"""Utilities for mapping between actual and formal arguments (and their types)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

from mypy import nodes
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2
from mypy.tuple_normal_form import TupleHelper, TupleNormalForm
from mypy.types import (
    AnyType,
    Instance,
    ParamSpecType,
    TupleType,
    Type,
    TypedDictType,
    TypeOfAny,
    UninhabitedType,
    UnpackType,
    get_proper_type,
)

if TYPE_CHECKING:
    from mypy.infer import ArgumentInferContext


def map_actuals_to_formals(
    actual_kinds: list[nodes.ArgKind],
    actual_names: Sequence[str | None] | None,
    formal_kinds: list[nodes.ArgKind],
    formal_names: Sequence[str | None],
    actual_arg_type: Callable[[int], Type],
) -> list[list[int]]:
    """Calculate mapping between actual (caller) args and formals.

    The result contains a list of caller argument indexes mapping to each
    callee argument index, indexed by callee index.

    The caller_arg_type argument should evaluate to the type of the actual
    argument type with the given index.
    """
    nformals = len(formal_kinds)
    formal_to_actual: list[list[int]] = [[] for i in range(nformals)]
    ambiguous_actual_kwargs: list[int] = []
    fi = 0
    for ai, actual_kind in enumerate(actual_kinds):
        if actual_kind == ARG_POS:
            if fi < nformals:
                if formal_kinds[fi] in (ARG_POS, ARG_OPT):
                    formal_to_actual[fi].append(ai)
                    fi += 1
                elif formal_kinds[fi] == ARG_STAR:
                    formal_to_actual[fi].append(ai)
        elif actual_kind == ARG_STAR:
            # convert the actual argument type to a tuple-like type
            star_arg_type = TupleNormalForm.from_star_arg(actual_arg_type(ai))

            # for a variadic argument use a negative value, so it remains truthy when decremented
            # otherwise, use the length of the prefix.
            num_actual_items = -1 if star_arg_type.is_variadic else len(star_arg_type.prefix)
            # note: empty tuple star-args will not get mapped to anything
            while fi < nformals and num_actual_items:
                if formal_kinds[fi] in (ARG_POS, ARG_OPT, ARG_STAR):
                    formal_to_actual[fi].append(ai)
                    num_actual_items -= 1
                if formal_kinds[fi] in (ARG_STAR, ARG_NAMED, ARG_NAMED_OPT, ARG_STAR2):
                    break
                fi += 1
        elif actual_kind.is_named():
            assert actual_names is not None, "Internal error: named kinds without names given"
            name = actual_names[ai]
            if name in formal_names and formal_kinds[formal_names.index(name)] != ARG_STAR:
                formal_to_actual[formal_names.index(name)].append(ai)
            elif ARG_STAR2 in formal_kinds:
                formal_to_actual[formal_kinds.index(ARG_STAR2)].append(ai)
        else:
            assert actual_kind == ARG_STAR2
            actualt = get_proper_type(actual_arg_type(ai))
            if isinstance(actualt, TypedDictType):
                for name in actualt.items:
                    if name in formal_names:
                        formal_to_actual[formal_names.index(name)].append(ai)
                    elif ARG_STAR2 in formal_kinds:
                        formal_to_actual[formal_kinds.index(ARG_STAR2)].append(ai)
            else:
                # We don't exactly know which **kwargs are provided by the
                # caller, so we'll defer until all the other unambiguous
                # actuals have been processed
                ambiguous_actual_kwargs.append(ai)

    if ambiguous_actual_kwargs:
        # Assume the ambiguous kwargs will fill the remaining arguments.
        #
        # TODO: If there are also tuple varargs, we might be missing some potential
        #       matches if the tuple was short enough to not match everything.
        unmatched_formals = [
            fi
            for fi in range(nformals)
            if (
                formal_names[fi]
                and (not formal_to_actual[fi] or actual_kinds[formal_to_actual[fi][0]] == ARG_STAR)
                and formal_kinds[fi] != ARG_STAR
            )
            or formal_kinds[fi] == ARG_STAR2
        ]
        for ai in ambiguous_actual_kwargs:
            for fi in unmatched_formals:
                formal_to_actual[fi].append(ai)

    return formal_to_actual


def map_formals_to_actuals(
    actual_kinds: list[nodes.ArgKind],
    actual_names: Sequence[str | None] | None,
    formal_kinds: list[nodes.ArgKind],
    formal_names: list[str | None],
    actual_arg_type: Callable[[int], Type],
) -> list[list[int]]:
    """Calculate the reverse mapping of map_actuals_to_formals."""
    formal_to_actual = map_actuals_to_formals(
        actual_kinds, actual_names, formal_kinds, formal_names, actual_arg_type
    )
    # Now reverse the mapping.
    actual_to_formal: list[list[int]] = [[] for _ in actual_kinds]
    for formal, actuals in enumerate(formal_to_actual):
        for actual in actuals:
            actual_to_formal[actual].append(formal)
    return actual_to_formal


class ArgTypeExpander:
    """Utility class for mapping actual argument types to formal arguments.

    One of the main responsibilities is to expand caller tuple *args and TypedDict
    **kwargs, and to keep track of which tuple/TypedDict items have already been
    consumed.

    Example:

       def f(x: int, *args: str) -> None: ...
       f(*(1, 'x', 1.1))

    We'd call expand_actual_type three times:

      1. The first call would provide 'int' as the actual type of 'x' (from '1').
      2. The second call would provide 'str' as one of the actual types for '*args'.
      2. The third call would provide 'float' as one of the actual types for '*args'.

    A single instance can process all the arguments for a single call. Each call
    needs a separate instance since instances have per-call state.
    """

    def __init__(self, context: ArgumentInferContext) -> None:
        # Next tuple *args index to use.
        self.tuple_index = 0
        # Keyword arguments in TypedDict **kwargs used.
        self.kwargs_used: set[str] | None = None
        # Type context for `*` and `**` arg kinds.
        self.context = context

    def expand_actual_type(
        self,
        actual_type: Type,
        actual_kind: nodes.ArgKind,
        formal_name: str | None,
        formal_kind: nodes.ArgKind,
    ) -> Type:
        """Return the actual (caller) type(s) of a formal argument with the given kinds.

        If the actual argument is a tuple *args, then:
            1. If the formal argument is positional, return the next individual tuple item that
               maps to the formal arg. If the tuple is exhausted, return 'None'
            2. If the formal argument is also *args, returns a tuple type with the items
               that map to the formal arg.
               If the tuple is exhausted, returns an empty tuple type.

        If the actual argument is a TypedDict **kwargs, return the next matching typed dict
        value type based on formal argument name and kind.

        This is supposed to be called for each formal, in order. Call multiple times per
        formal if multiple actuals map to a formal.
        """
        original_actual = actual_type
        actual_type = get_proper_type(actual_type)

        if actual_kind == ARG_POS:
            if formal_kind in (ARG_POS, ARG_OPT):
                # return as-is
                return original_actual
            elif formal_kind == ARG_STAR:
                # wrap in a tuple
                return original_actual
                # return TupleType([original_actual], fallback=self.context.tuple_type)
            else:
                assert False, f"unexpected formal kind {formal_kind} for positional actual"

        elif actual_kind == ARG_STAR:
            # parse *args into a TupleType.
            star_args_type = self.parse_star_argument(actual_type)
            tuple_helper = TupleHelper(self.context.tuple_typeinfo)

            # star_args_type failed to parse. treat as if it were tuple[Any, ...]
            if isinstance(star_args_type, AnyType):
                any_tuple = self.context.make_tuple_instance_type(AnyType(TypeOfAny.from_error))
                star_args_type = self.context.make_tuple_type([UnpackType(any_tuple)])

            assert isinstance(star_args_type, TupleType)

            # we are mapping an actual *args to positional arguments.
            if formal_kind in (ARG_POS, ARG_OPT):
                value = tuple_helper.get_item(star_args_type, self.tuple_index)
                self.tuple_index += 1

                # FIXME: In principle, None should indicate out-of-bounds access
                #   caused by an error in formal_to_actual mapping.
                # assert value is not None, "error in formal_to_actual mapping"
                # However, in some cases due to lack of machinery it can happen:
                # For example f(*[]). Then formal_to_actual is ignorant of the fact
                # that the list is empty, but when materializing the tuple we actually get an empty tuple.
                # Therefore, we currently just return UninhabitedType in this case.
                value = UninhabitedType() if value is None else value

                # if the argument is exhausted, reset the index
                if not star_args_type.is_variadic and self.tuple_index >= len(
                    star_args_type.proper_items
                ):
                    self.tuple_index = 0
                return value

            # we are mapping an actual *args input to a *args formal argument.
            elif formal_kind == ARG_STAR:
                # get the slice from the current index to the end of the tuple.
                r = tuple_helper.get_slice(star_args_type, self.tuple_index, None)
                # r = star_args_type.slice(
                #     self.tuple_index, None, None, fallback=self.context.tuple_type
                # )
                self.tuple_index = 0
                # assert r is not None, f"failed to slice {star_args_type} at {self.tuple_index}"
                return r

            else:
                raise ValueError(f"Unexpected formal kind {formal_kind} for *args")

        elif actual_kind == ARG_NAMED:
            return original_actual

        elif actual_kind == ARG_STAR2:
            from mypy.subtypes import is_subtype

            if isinstance(actual_type, TypedDictType):
                if self.kwargs_used is None:
                    self.kwargs_used = set()
                if formal_kind != ARG_STAR2 and formal_name in actual_type.items:
                    # Lookup type based on keyword argument name.
                    assert formal_name is not None
                else:
                    # Pick an arbitrary item if no specified keyword is expected.
                    formal_name = (set(actual_type.items.keys()) - self.kwargs_used).pop()
                self.kwargs_used.add(formal_name)
                return actual_type.items[formal_name]
            elif isinstance(actual_type, Instance) and is_subtype(
                actual_type, self.context.mapping_type
            ):
                # Only `Mapping` type can be unpacked with `**`.
                # Other types will produce an error somewhere else.
                return map_instance_to_supertype(actual_type, self.context.mapping_type.type).args[
                    1
                ]
            elif isinstance(actual_type, ParamSpecType):
                # ParamSpec is valid in **kwargs but it can't be unpacked.
                return actual_type
            else:
                return AnyType(TypeOfAny.from_error)

        else:
            assert False, f"unexpected actual kind {actual_kind}"

    def parse_star_argument(self, star_arg: Type, /) -> TupleType:
        r"""Parse the type of ``*args`` argument into a tuple type.

        Note: For star parameters, use `parse_star_parameter` instead.
        """
        tnf = TupleNormalForm.from_star_arg(star_arg)
        return tnf.materialize(self.context)

    def parse_star_parameter(self, star_param: Type, /) -> TupleType:
        r"""Parse the type of a ``*args: T`` annotation into a tuple type.

        Note: For star arguments, use `parse_star_argument` instead.

        Note: mypy in an earlier analysis phase wraps converts these annotations:

        Examples:
            - ``*args: int`` --> ``int`` --> ``Instance(tuple_type, [int])``
            - ``*args: P.args`` --> ``ParamSpecType`` --> ``TupleType[Unpack[ParamSpecType]]``
            - ``*args: *tuple[int, int]`` --> ``UnpackType[TupleType[[int, int]]]`` --> ``TupleType[int, int]``
            - ``*args: *Ts`` --> ``UnpackType[TypeVarTupleType]`` --> ``TupleType[Unpack[TypeVarTupleType]]``
            - ``*args: *tuple[int, ...]`` --> ``UnpackType[TupleType[[int, ...]]]`` --> ``TupleType[Unpack[Instance(tuple_type, [int])]]``
        """
        p_t = get_proper_type(star_param)
        if isinstance(p_t, UnpackType):
            unpacked = get_proper_type(p_t.type)
            if isinstance(unpacked, TupleType):
                return unpacked
            return TupleType([p_t], fallback=self.context.fallback_tuple)

        elif isinstance(p_t, ParamSpecType):
            # we put the ParamSpecType here inside
            parsed = UnpackType(p_t)
            return TupleType([parsed], fallback=self.context.fallback_tuple)

        else:  # e.g. *args: int  --> *args: *tuple[int, ...]
            parsed = UnpackType(self.context.make_tuple_instance_type(p_t))
            return TupleType([parsed], fallback=self.context.fallback_tuple)

    def expand_all_actuals_and_formal_star_arg(self, actual_types, actual_kinds, formal_type):
        r"""
        option 1 iterate over the formal_to_actual
        option 2 iterate over the actuals, and keep track of which formal we are at.
        option 3 iterate over the formals, and keep track of which actual we are at.
        in any case we also need like 1 tuple index per actual, although it only
        really matters for the critical argument.
        may it is better to rewrite the errors:
        argument X to fn has incompatible type, got X but expected Y (POS to POS
        argument X to fn has incompatible type, got *(....) but expected Y (STAR to POS)
        argument X to fn has incompatible type, got X but expected *(...) (POS to STAR)
        argument X to fn has incompatible type, got *(...) but expected *(...) (STAR to STAR)

        IMPORTANT: on the callee side, at most ARG_STAR can be variadic (but it doesn't have to)
        in this case, we know all variadic positional arguments must be mapped to this single ARG_STAR
        the leading callers STAR_ARG can also be mapped to a number of preceding positionals arguments.

        case: the formal definition is bounded and the actual definition is bonded:
        represent the formal in TNF as tuple[T1, ..., Tk]
        represent the actual in TNF as tuple[P1, ..., Pn]
        if n > k, raise a too many arguments error
        if n < k, raise a too few arguments error
        compare the first min(n, k) items one by one.

        case: the formal definition is bounded, but the actual arguments are unbounded.
        represent formal in TNF as tuple[T1, ..., Tk]
        represent actual in TNF as tuple[P1, ..., Pn, *Vs, Q1, ..., Qm]
        Then in n+m > k, raise a too many arguments error.
        otherwise, compare the first n-many items to the prefix (actually, min(n,k)-many)
                 , the last m-many items to the suffix (actually min(m, k-min(n,k))-many)
                 , and all other items to the unpacked *Vs.

        case: the formal definition is unbounded, but the actual definition is bounded:
        represent formal in TNF as tuple[P1, ..., Pn, *Vs, S1, ..., Sm]
        represent actual in TNF as tuple[T1, ..., Tk]
        Then in k < n+m, raise a too few arguments error.
        otherwise, compare the first n-many items to the prefix
                 , the last m-many items to the suffix
                 , and all other items to the unpacked *Vs.

        case: both the formal definition and the actual definition are unbounded:
        represent formal in TNF as tuple[P1, ..., Pn, *Vs, S1, ..., Sm]
        represent actual in TNF as tuple[T1, ..., Tk, *Us, Q1, ..., Ql]
        Then:
        - compare the first min(n, k) many items one by one.
        - compare the last min(m, l) many items one by one.

        For the remaining items, there are 4 cases:
        formal                                     actual
        tuple[P1, ..., Pn, *Vs, S1, ..., Sm]  and  tuple[*Us]
        tuple[*Vs, S1, ..., Sm]               and  tuple[T1, ..., Tn, *Us]
        tuple[P1, ..., Pn, *Vs]               and  tuple[*Us, Q1, ..., Qm]
        tuple[*Vs]                            and  tuple[T1, ..., Tn, *Us, Q1, ..., Qm]
        check these according to the rules above.
        """
