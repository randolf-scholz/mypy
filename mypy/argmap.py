"""Utilities for mapping between actual and formal arguments (and their types)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable
from typing_extensions import assert_never

from mypy import nodes
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2
from mypy.tuple_normal_form import FlatTuple, TupleNormalForm
from mypy.types import (
    AnyType,
    Instance,
    ParamSpecType,
    TupleType,
    Type,
    TypedDictType,
    TypeOfAny,
    TypeVarTupleType,
    UninhabitedType,
    UnpackType,
    find_unpack_in_list,
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
        if actual_kind == nodes.ARG_POS:
            if fi < nformals:
                if not formal_kinds[fi].is_star():
                    formal_to_actual[fi].append(ai)
                    fi += 1
                elif formal_kinds[fi] == nodes.ARG_STAR:
                    formal_to_actual[fi].append(ai)
        elif actual_kind == nodes.ARG_STAR:
            # convert the actual argument type to a tuple-like type
            star_arg_type = TupleNormalForm.from_star_arg(actual_arg_type(ai))

            # for a variadic argument use a negative value, so it remains truthy when decremented
            # otherwise, use the length of the prefix.
            num_actual_items = -1 if star_arg_type.variadic else len(star_arg_type.prefix)

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
            if name in formal_names and formal_kinds[formal_names.index(name)] != nodes.ARG_STAR:
                formal_to_actual[formal_names.index(name)].append(ai)
            elif nodes.ARG_STAR2 in formal_kinds:
                formal_to_actual[formal_kinds.index(nodes.ARG_STAR2)].append(ai)
        else:
            assert actual_kind == nodes.ARG_STAR2
            actualt = get_proper_type(actual_arg_type(ai))
            if isinstance(actualt, TypedDictType):
                for name in actualt.items:
                    if name in formal_names:
                        formal_to_actual[formal_names.index(name)].append(ai)
                    elif nodes.ARG_STAR2 in formal_kinds:
                        formal_to_actual[formal_kinds.index(nodes.ARG_STAR2)].append(ai)
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
                and (
                    not formal_to_actual[fi]
                    or actual_kinds[formal_to_actual[fi][0]] == nodes.ARG_STAR
                )
                and formal_kinds[fi] != nodes.ARG_STAR
            )
            or formal_kinds[fi] == nodes.ARG_STAR2
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
    ) -> Type | None:
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
        if actual_kind == ARG_STAR:
            # parse *args as one of the following:
            #    TupleType | ParamSpecType | AnyType
            # Then, depending on the formal type, return a tuple-like type or an item from the tuple.
            star_args_type = parse_star_args_type(actual_type, self.context)

            # fast path: star_args_type failed to parse, return AnyType
            if isinstance(star_args_type, AnyType):
                return star_args_type

            # otherwise, this must be a TupleType
            assert isinstance(star_args_type, TupleType)


            # we are mapping an actual *args input to a *args formal argument.
            if formal_kind == ARG_STAR:
                # get the slice from the current index to the end of the tuple.
                r = star_args_type.slice(
                    self.tuple_index, None, None, fallback=self.context.tuple_type
                )
                assert r is not None, f"failed to slice {star_args_type} at {self.tuple_index}"
                return r

            # we are mapping an actual *args to positional arguments.
            elif formal_kind in (ARG_POS, ARG_OPT):
                self.tuple_index += 1
                return self.context.get_tuple_item(star_args_type, self.tuple_index -1)


                if isinstance(star_args_type, TupleType):

                    self.tuple_index += 1
                    if item is None:
                        # out of bounds!
                        self.tuple_index = 0

                    if item is None:
                        # out of bounds
                        return None

                    if self.tuple_index >= len(star_args_type.items):
                        # out of bounds
                        return None

                    item = star_args_type.items[self.tuple_index]
                    p_t = get_proper_type(item)
                    if isinstance(p_t, UnpackType):
                        unpacked = get_proper_type(p_t.type)
                        if isinstance(unpacked, TypeVarTupleType):
                            return unpacked.tuple_fallback.args[0]
                        elif self.context.is_tuple_instance_type(unpacked):
                            # If the unpacked type is a tuple, return the first item.
                            return unpacked.args[0]
                        else:
                            raise TypeError(
                                f"Unexpected unpacked type {unpacked} in expanded *args type."
                                f"\n\toriginal type: {original_actual}"
                                f"\n\texpanded type: {star_args_type}"
                            )
                    else:
                        self.tuple_index += 1
                        if self.tuple_index >= len(star_args_type.items):
                            # tuple is exhausted, reset the index.
                            self.tuple_index = 0
                        return p_t

                elif isinstance(star_args_type, ParamSpecType | AnyType):
                    return star_args_type
                else:
                    assert_never(star_args_type)
            else:
                raise ValueError(f"Unexpected formal kind {formal_kind} for *args")
        elif actual_kind == nodes.ARG_STAR2:
            from mypy.subtypes import is_subtype

            if isinstance(actual_type, TypedDictType):
                if self.kwargs_used is None:
                    self.kwargs_used = set()
                if formal_kind != nodes.ARG_STAR2 and formal_name in actual_type.items:
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
            # No translation for other kinds -- 1:1 mapping.
            return original_actual

    def _get_tuple_item(self, tt: TupleType, index: int) -> Type:
        r"""Get the predicted type of indexing the tuple.

        Assuming the tuple type is in Tuple Normal Form, then the result is:

        tuple[P1, ..., Pn, *Vs, S1, ..., Sm] @ index =
            Iterable_type[Vs]  if index < -m
            S[index]           if -m ≤ index < 0
            P[index]           if 0 ≤ index < n
            Iterable_type[Vs]  if index ≥ n

        If the tuple has no variadic part, the it works just like regular indexing
        on the proper items
        """
        proper_items = tt.proper_items()
        unpack_index = find_unpack_in_list(proper_items)

        if unpack_index is None:
            return proper_items[index]

        n = unpack_index - 1
        m = len(proper_items) - unpack_index

        if -m <= index < n:
            return proper_items[index]

        item = proper_items[unpack_index]
        assert isinstance(item, UnpackType)
        unpacked = get_proper_type(item.type)

        # convert to Iterable[T] and return T

    def parse_star_args_type(self, typ: Type) -> FlatTuple | ParamSpecType | AnyType:
        r"""Parse the type of ``*args`` argument into a tuple or ParamSpecType type."""
        p_t = get_proper_type(typ)
        if isinstance(p_t, ParamSpecType | AnyType):
            # just return the type as-is
            return p_t


def parse_star_args_type(
    typ: Type, /, context: ArgumentInferContext
) -> FlatTuple | AnyType:
    p_t = get_proper_type(typ)
    if isinstance(p_t, AnyType):
        # just return the type as-is
        return p_t

    tnf = TupleNormalForm.from_star_arg(p_t)
    return tnf.materialize(context)

    # def _solve_as_iterable(self, typ: Type) -> IterableType | AnyType:
    #     r"""Use the solver to cast a type as Iterable[T].
    #
    #     Returns `AnyType` if solving fails.
    #     """
    #     from mypy.constraints import infer_constraints_for_callable
    #     from mypy.nodes import ARG_POS
    #     from mypy.solve import solve_constraints
    #
    #     # We first create an upcast function:
    #     #    def [T] (Iterable[T]) -> Iterable[T]: ...
    #     # and then solve for T, given the input type as the argument.
    #     T = TypeVarType(
    #         "T",
    #         "T",
    #         TypeVarId(-1),
    #         values=[],
    #         upper_bound=AnyType(TypeOfAny.from_omitted_generics),
    #         default=AnyType(TypeOfAny.from_omitted_generics),
    #     )
    #     target = self.context.make_iterable_instance_type(T)
    #     upcast_callable = CallableType(
    #         variables=[T],
    #         arg_types=[target],
    #         arg_kinds=[ARG_POS],
    #         arg_names=[None],
    #         ret_type=target,
    #         fallback=self.context.function_type,
    #     )
    #     constraints = infer_constraints_for_callable(
    #         upcast_callable, [typ], [ARG_POS], [None], [[0]], self.context
    #     )
    #
    #     (sol,), _ = solve_constraints([T], constraints)
    #
    #     if sol is None:  # solving failed, return AnyType fallback
    #         return AnyType(TypeOfAny.from_error)
    #     return self.context.make_iterable_instance_type(sol)

    # def as_iterable_type(self, typ: Type) -> IterableType | AnyType:
    #     """Reinterpret a type as Iterable[T], or return AnyType if not possible.
    #
    #     This function specially handles certain types like UnionType, TupleType, and UnpackType.
    #     Otherwise, the upcasting is performed using the solver.
    #     """
    #     p_t = get_proper_type(typ)
    #     if self.context.is_iterable_instance_type(p_t) or isinstance(p_t, AnyType):
    #         return p_t
    #     elif isinstance(p_t, UnionType):
    #         # If the type is a union, map each item to the iterable supertype.
    #         # the return the combined iterable type Iterable[A] | Iterable[B] -> Iterable[A | B]
    #         converted_types = [self.as_iterable_type(get_proper_type(item)) for item in p_t.items]
    #
    #         if any(not self.context.is_iterable_instance_type(it) for it in converted_types):
    #             # if any item could not be interpreted as Iterable[T], we return AnyType
    #             return AnyType(TypeOfAny.from_error)
    #         else:
    #             # all items are iterable, return Iterable[T₁ | T₂ | ... | Tₙ]
    #             iterable_types = cast("list[IterableType]", converted_types)
    #             arg = make_simplified_union([it.args[0] for it in iterable_types])
    #             return self.context.make_iterable_instance_type(arg)
    #     elif isinstance(p_t, TupleType):
    #         # maps tuple[A, B, C] -> Iterable[A | B | C]
    #         # note: proper_elements may contain UnpackType, for instance with
    #         #   tuple[None, *tuple[None, ...]]..
    #         proper_elements = [get_proper_type(t) for t in flatten_nested_tuples(p_t.items)]
    #         args: list[Type] = []
    #         for p_e in proper_elements:
    #             if isinstance(p_e, UnpackType):
    #                 r = self.as_iterable_type(p_e)
    #                 if self.context.is_iterable_instance_type(r):
    #                     args.append(r.args[0])
    #                 else:
    #                     # this *should* never happen, since UnpackType should
    #                     # only contain TypeVarTuple or a variable length tuple.
    #                     # However, we could get an `AnyType(TypeOfAny.from_error)`
    #                     # if for some reason the solver was triggered and failed.
    #                     args.append(r)
    #             else:
    #                 args.append(p_e)
    #         return self.context.make_iterable_instance_type(make_simplified_union(args))
    #     elif isinstance(p_t, UnpackType):
    #         return self.as_iterable_type(p_t.type)
    #     elif isinstance(p_t, (TypeVarType, TypeVarTupleType)):
    #         return self.as_iterable_type(p_t.upper_bound)
    #     elif self.context.is_iterable(p_t):
    #         # TODO: add a 'fast path' (needs measurement) that uses the map_instance_to_supertype
    #         #   mechanism? (Only if it works: gh-19662)
    #         return self._solve_as_iterable(p_t)
    #     return AnyType(TypeOfAny.from_error)

    # def parse_star_args_type(self, typ: Type) -> FlatTuple | ParamSpecType | AnyType:
    #     """Parse the type of ``*args`` argument into a tuple or ParamSpecType type.
    #
    #     Examples:
    #         tuple[int, int]       -> tuple[int, int]
    #         list[int]             -> tuple[*tuple[int, ...]]
    #         list[int] | list[str] -> tuple[*tuple[int | str, ...]]
    #         Ts                    -> tuple[*Ts]
    #         P.args                -> P.args
    #         Any                   -> Any
    #
    #     Also returns `Any` if the type cannot be parsed or is invalid.
    #     """
    #     p_t = get_proper_type(typ)
    #     if isinstance(p_t, ParamSpecType | AnyType):
    #         # just return the type as-is
    #         return p_t
    #     elif isinstance(p_t, TupleType):
    #         # ensure the tuple is flattened.
    #         flat_tuple = TupleType(flatten_nested_tuples(p_t.items), fallback=p_t.partial_fallback)
    #         return cast(FlatTuple, flat_tuple)
    #     elif isinstance(p_t, TypeVarTupleType):
    #         flat_tuple = TupleType([UnpackType(p_t)], fallback=p_t.tuple_fallback)
    #         return cast(FlatTuple, flat_tuple)
    #     elif isinstance(p_t, UnionType):
    #         proper_items = [get_proper_type(t) for t in p_t.items]
    #         # consider 2 cases:
    #
    #         # 1. Union of tuple -> tuple
    #         if all(isinstance(t, TupleType) for t in proper_items):
    #             proper_items = cast("list[TupleType]", proper_items)
    #             return self.combine_tuple_types(proper_items)
    #         # 2. Union of iterable types, e.g. Iterable[A] | Iterable[B]
    #         #    In this case return tuple[*tuple[A | B, ...]]
    #         else:
    #             converted_types = [self.as_iterable_type(p_i) for p_i in proper_items]
    #             if all(self.context.is_iterable_instance_type(it) for it in converted_types):
    #                 # all items are iterable, return tuple[T1 | T2 | ... | Tn, ...]
    #                 iterables = cast("list[IterableType]", converted_types)
    #                 arg = make_simplified_union([it.args[0] for it in iterables])
    #                 inner_tuple = self.context.make_tuple_instance_type(arg)
    #                 return TupleType([UnpackType(inner_tuple)], fallback=inner_tuple)
    #             else:
    #                 # some items in the union are not iterable, return AnyType
    #                 error_type = AnyType(TypeOfAny.from_error)
    #                 return error_type
    #     else:
    #         parsed = self.as_iterable_type(p_t)
    #         if self.context.is_iterable_instance_type(parsed):
    #             inner_tuple = self.context.make_tuple_instance_type(parsed.args[0])
    #             return TupleType([UnpackType(inner_tuple)], fallback=inner_tuple)
    #         # the argument is not iterable
    #         error_type = AnyType(TypeOfAny.from_error)
    #         return error_type

    # def parse_star_args_type(self, typ: Type) -> FlatTuple | ParamSpecType | AnyType:
    #     """Parse the type of ``*args`` argument into a tuple or ParamSpecType type.
    #
    #     Examples:
    #         tuple[int, int]       -> tuple[int, int]
    #         list[int]             -> tuple[*tuple[int, ...]]
    #         list[int] | list[str] -> tuple[*tuple[int | str, ...]]
    #         Ts                    -> tuple[*Ts]
    #         P.args                -> P.args
    #         Any                   -> Any
    #
    #     Also returns `Any` if the type cannot be parsed or is invalid.
    #     """
    #     p_t = get_proper_type(typ)
    #     if isinstance(p_t, ParamSpecType | AnyType):
    #         # just return the type as-is
    #         return p_t
    #     elif isinstance(p_t, TupleType):
    #         # ensure the tuple is flattened.
    #         flat_tuple = TupleType(flatten_nested_tuples(p_t.items), fallback=p_t.partial_fallback)
    #         return cast(FlatTuple, flat_tuple)
    #     elif isinstance(p_t, TypeVarTupleType):
    #         flat_tuple = TupleType([UnpackType(p_t)], fallback=p_t.tuple_fallback)
    #         return cast(FlatTuple, flat_tuple)
    #     elif isinstance(p_t, UnionType):
    #         proper_items = [get_proper_type(t) for t in p_t.items]
    #         # consider 2 cases:
    #
    #         # 1. Union of tuple -> tuple
    #         if all(isinstance(t, TupleType) for t in proper_items):
    #             proper_items = cast("list[TupleType]", proper_items)
    #             return self.combine_tuple_types(proper_items)
    #         # 2. Union of iterable types, e.g. Iterable[A] | Iterable[B]
    #         #    In this case return tuple[*tuple[A | B, ...]]
    #         else:
    #             converted_types = [self.as_iterable_type(p_i) for p_i in proper_items]
    #             if all(self.context.is_iterable_instance_type(it) for it in converted_types):
    #                 # all items are iterable, return tuple[T1 | T2 | ... | Tn, ...]
    #                 iterables = cast("list[IterableType]", converted_types)
    #                 arg = make_simplified_union([it.args[0] for it in iterables])
    #                 inner_tuple = self.context.make_tuple_instance_type(arg)
    #                 return TupleType([UnpackType(inner_tuple)], fallback=inner_tuple)
    #             else:
    #                 # some items in the union are not iterable, return AnyType
    #                 error_type = AnyType(TypeOfAny.from_error)
    #                 return error_type
    #     else:
    #         parsed = self.as_iterable_type(p_t)
    #         if self.context.is_iterable_instance_type(parsed):
    #             inner_tuple = self.context.make_tuple_instance_type(parsed.args[0])
    #             return TupleType([UnpackType(inner_tuple)], fallback=inner_tuple)
    #         # the argument is not iterable
    #         error_type = AnyType(TypeOfAny.from_error)
    #         return error_type

    # def combine_tuple_types(self, types: Sequence[TupleType]) -> FlatTuple:
    #     head, body, tail = split_union_of_tuple_types(types)
    #     fallback = self.context.tuple_type
    #     if body:
    #         # upcast the body part to Iterable[T].
    #         virtual_tuple = TupleType(body, fallback=fallback)
    #         virtual_iterable = self.as_iterable_type(virtual_tuple)
    #         if self.context.is_iterable_instance_type(virtual_iterable):
    #             body_arg = virtual_iterable.args[0]
    #         elif isinstance(virtual_iterable, AnyType):
    #             body_arg = virtual_iterable
    #         else:
    #             assert_never(virtual_iterable)
    #         variable_part = self.context.make_tuple_instance_type(body_arg)
    #         flat_tuple = TupleType([*head, UnpackType(variable_part), *tail], fallback=fallback)
    #         return cast(FlatTuple, flat_tuple)
    #     # there are neither tail nor body items, so we return just the head part
    #     assert not tail
    #     result = TupleType(head, fallback=fallback)
    #     return cast(FlatTuple, result)

    # def _combine_tuple_types(self, types: Sequence[TupleType]) -> TupleType:
    #     """Combine multiple tuple types into a single tuple type.
    #
    #     This creates an upper bound for the union of the input tuple types.
    #     If all input tuples are of the same (real) size, so is the result.
    #
    #     Examples:
    #         tuple[int, int], tuple[None, None]
    #             -> tuple[int | None, int | None]
    #
    #         tuple[int, *tuple[int, ...], int],
    #         tuple[None, *tuple[None, ...], None]
    #             -> tuple[int | None, *tuple[int | None, ...], int | None]
    #
    #         tuple[int, *tuple[int, ...], str, *tuple[str, ...], int]
    #         tuple[None, *tuple[None, ...], None]
    #             -> tuple[int | None, *tuple[int | str | None, ...], int | None]
    #
    #     Note:
    #         According to the type spec at the time of writing, only one unbounded tuple
    #         is allowed in a tuple type, but this code should work even with multiple
    #         unbounded tuples, e.g. `tuple[*tuple[None, ...], str, *tuple[None, ...]]`
    #         See: https://typing.python.org/en/latest/spec/tuples.html#unpacked-tuple-form
    #     """
    #     heads: list[list[Type]]
    #     bodies: list[list[Type]]
    #     tails: list[list[Type]]
    #     remaining_head_items: list[list[Type]]
    #     remaining_tail_items: list[list[Type]]
    #     target_head_items: list[Type] = []
    #     target_body_items: list[Type] = []
    #     target_tail_items: list[Type] = []
    #
    #     # split each tuple
    #     heads, bodies, tails = zip(*(split_tuple_type(typ) for typ in types))
    #
    #     # 1. process all heads in parallel, stopping when one of the heads is exhausted
    #     shared_head_length = min(len(head) for head in heads)
    #     for items in zip(*(head[:shared_head_length] for head in heads)):
    #         # append the union of the items to the head part
    #         target_head_items.append(make_simplified_union(items))
    #     # collect all the remaining head items from generators that were not exhausted
    #     remaining_head_items = [head[shared_head_length:] for head in heads]
    #
    #     # If a tuple has no body items, prepend the remaining head items to the tail.
    #     # This addresses cases like combining `tuple[A, B, C]` with `tuple[X, *tuple[Y, ...], Z]`.
    #     # which should yield tuple[A | X, *tuple[B | Y, ...], C | Z]
    #     for remaining_head, body, tail in zip(remaining_head_items, bodies, tails):
    #         if not body:
    #             # move all remaining head items to the start of the tail
    #             _tail = tail[:]
    #             tail.clear()
    #             tail.extend(remaining_head)
    #             tail.extend(_tail)
    #             remaining_head.clear()
    #
    #     # 2. process all tails in parallel, in reverse, stopping when one of the tails is exhausted
    #     shared_tail_length = min(len(tail) for tail in tails)
    #     for items in zip(*(tail[-1 : -shared_tail_length - 1 : -1] for tail in tails)):
    #         # append the union of the items to the tail part
    #         target_tail_items.append(make_simplified_union(items))
    #     # collect all the remaining tail items from generators that were not exhausted
    #     target_tail_items.reverse()  # reverse to maintain original order
    #     remaining_tail_items = [tail[: len(tail) - shared_tail_length] for tail in tails]
    #     # note: do not use tail[:-shared_tail_length]; breaks when shared_tail_length=0
    #
    #     # 3. process all bodies
    #     for body in bodies:
    #         for item in body:
    #             p_t = get_proper_type(item)
    #             if isinstance(p_t, UnpackType):
    #                 unpacked = get_proper_type(p_t.type)
    #                 if isinstance(unpacked, TypeVarTupleType):
    #                     item = unpacked.tuple_fallback
    #                     target_body_items.append(item)
    #                 else:
    #                     assert (
    #                         isinstance(unpacked, Instance)
    #                         and unpacked.type.fullname == "builtins.tuple"
    #                     )
    #                     item = unpacked.args[0]
    #                     target_body_items.append(item)
    #             else:
    #                 target_body_items.append(item)
    #
    #     # 4. collected all items that will be put into the variable part
    #     combined_items = [
    #         *chain.from_iterable(remaining_head_items),
    #         *target_body_items,
    #         *chain.from_iterable(remaining_tail_items),
    #     ]
    #     fallback = self.context.tuple_type
    #
    #     if combined_items:
    #         variable_type = make_simplified_union(combined_items)
    #         variable_part = self.context.make_tuple_instance_type(variable_type)
    #         return TupleType(
    #             [*target_head_items, UnpackType(variable_part), *target_tail_items],
    #             fallback=fallback,
    #         )
    #     # there are neither tail nor body items, so we return just the head part
    #     assert not target_tail_items
    #     return TupleType(target_head_items, fallback=fallback)

    # def combine_finite_tuple_types(self, types: Sequence[FiniteTuple]) -> TupleType:
    #     """Combine multiple finite tuple types into a single tuple type.
    #
    #     The result is an upper bound for the union of the input tuple types.
    #     If all input tuples are of the same size, so is the result.
    #     If any input tuples have different sizes, the result will be a variable-length tuple.
    #
    #     Note:
    #         We treat each input tuple as consisting of two parts:
    #
    #         1. The head part, which are the first n items of the tuple, where n is the minimal length
    #         2. The tail part which are the remaining items of the tuple, if any.
    #
    #         The resulting tuple type will be of the form
    #
    #         tuple[T1, T2, ..., Tn, *tuple[U, ...]]
    #
    #         where each item `Tk` of the head part is the union of the item types of the input tuples,
    #         and `U` is the union of all the item types of all the tail parts of each input tuple.
    #
    #     Examples:
    #         tuple[int, int], tuple[None, None]           -> tuple[int | None, int | None]
    #         tuple[str], tuple[str, int]                  -> tuple[str, *tuple[int, ...]]
    #         tuple[str], tuple[str, str], tuple[int, int] -> tuple[str | int, *tuple[str | int, ...]]
    #     """
    #     if not types:
    #         raise ValueError("Expected at least one type, got empty sequence")
    #
    #     # 1. get the flattened elements of each tuple
    #     lengths = [get_real_tuple_length(tup) for tup in types]
    #     flattened_tuples: list[list[Type]] = [flatten_nested_tuples(typ.items) for typ in types]
    #     assert all(
    #         len(flat_items) == length for length, flat_items in zip(lengths, flattened_tuples)
    #     )
    #     # 2. compute the lengths
    #     minimal_length = min(lengths)
    #     maximal_length = max(lengths)
    #
    #     # 3. compute the head part by union-ing element-wise
    #     head_types = [
    #         make_simplified_union([tuple_items[i] for tuple_items in flattened_tuples])
    #         for i in range(minimal_length)
    #     ]
    #
    #     # if no tail part return early
    #     if minimal_length == maximal_length:
    #         fallback = self.context.make_tuple_instance_type(AnyType(TypeOfAny.unannotated))
    #         head_type = TupleType(head_types, fallback=fallback)
    #         return head_type
    #
    #     # 4. compute the tail part by union-ing all the remaining elements
    #     tail_type = make_simplified_union(
    #         [
    #             UnionType.make_union([*tuple_items[minimal_length:]])
    #             for tuple_items in flattened_tuples
    #         ]
    #     )
    #     # create the tail tuple type
    #     tail_tuple = self.context.make_tuple_instance_type(tail_type)
    #
    #     # return head part + tail part
    #     fallback = self.context.make_tuple_instance_type(
    #         make_simplified_union([*head_types, tail_type])
    #     )
    #     result = TupleType([*head_types, UnpackType(tail_tuple)], fallback=fallback)
    #     return result


# def flatten_finite_tuple_type(types: Iterable[Type]) -> list[Type]:
#     """Flatten a finite tuple type into a list of its items."""
#     res = []
#     for item in typ.items:
#         proper_item = get_proper_type(item)
#         if isinstance(proper_item, UnpackType):
#             unpacked = get_proper_type(proper_item.type)
#             if isinstance(unpacked, TypeVarTupleType):
#                 # If the unpacked type is a TypeVarTuple, we can use its upper bound.
#                 unpacked = get_proper_type(unpacked.upper_bound)
#                 res.extend(flatten_finite_tuple_type(unpacked))
#             elif is_finite_tuple_type(unpacked):
#                 # If the unpacked type is a TupleType, we can flatten it.
#                 res.extend(flatten_nested_tuples(unpacked.items))
#             elif is_tuple_instance_type(unpacked):
#                 # If the unpacked type is a tuple instance, we can flatten it.
#                 res.extend(flatten_nested_tuples(unpacked))
#             else:
#                 raise TypeError(f"Unexpected type for tuple: {typ}")
#         else:
#             res.append(item)
#     return res


# def normalize_finite_tuple(typ: Iterable[Type]) -> list[Type]:
#     r"""Get the length of a tuple type, or -1 if it is unbounded."""
#     p_t = get_proper_type(typ)
#     if isinstance(p_t, TypeVarTupleType):
#         # use the upper bound of the TypeVarTuple to determine the length
#         return get_real_tuple_length(p_t.upper_bound)
#     if is_tuple_instance_type(p_t):
#         # tuple[T, ...] is always unbounded, so return -1
#         return -1
#     if isinstance(p_t, TupleType):
#         size = 0
#         for item in p_t.items:
#             proper_item = get_proper_type(item)
#             if isinstance(proper_item, UnpackType):
#                 unpacked = get_proper_type(proper_item.type)
#                 assert is_tuple_like_type(unpacked)
#                 result = get_real_tuple_length(unpacked)
#                 if result == -1:
#                     return -1
#                 size += result
#             else:
#                 size += 1
#         return size
#     raise TypeError(f"Unexpected type for tuple length: {typ}")


# def real_tuple_length(typ: TupleLikeType) -> int:
#     if is_variable_tuple_type(typ):
#         return -1
#     if isinstance(typ, TypeVarTupleType):
#         bound = get_proper_type(typ.upper_bound)
#         assert isinstance(bound, TupleType | VariableTupleType) or is_variable_tuple_type(bound)
#         return real_tuple_length(bound)
#     if isinstance(typ, TupleType):
#     size = 0
#     for member in typ.items:
#         p_t = get_proper_type(member)
#         if isinstance(p_t, TypeVarTupleType):
#             bound = get_proper_type(p_t.upper_bound)
#             if isinstance(bound, TupleType):
#                 # If the TypeVarTuple has a bound that is a tuple, we can use its length.
#                 size += real_tuple_length(bound)
#             else:
#                 return -1
#         elif isinstance(p_t, UnpackType):
#             return -1
#         else:
#             size += 1
#         return size
#
#     raise TypeError(f"Expected TupleType or VariableTupleType, got {typ!r}")
