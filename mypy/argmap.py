"""Utilities for mapping between actual and formal arguments (and their types)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, cast
from typing_extensions import NewType, TypeAlias as _TypeAlias, TypeGuard, TypeIs

from mypy import nodes
from mypy.maptype import map_instance_to_supertype
from mypy.typeops import make_simplified_union
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    ParamSpecType,
    ProperType,
    TupleType,
    Type,
    TypedDictType,
    TypeOfAny,
    TypeVarId,
    TypeVarTupleType,
    TypeVarType,
    UnionType,
    UnpackType,
    flatten_nested_tuples,
    get_proper_type,
)

if TYPE_CHECKING:
    from mypy.infer import ArgumentInferContext, IterableType, TupleInstanceType


FiniteTupleType = NewType("FiniteTupleType", TupleType)
"""Represents an instance of `tuple[T1, T2, ..., Tn]` with a finite number of items."""
TupleLikeType: _TypeAlias = "TupleType | TypeVarTupleType | TupleInstanceType | FiniteTupleType"
r"""Types that are considered tuples or tuple-like."""


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
            # We need to know the actual type to map varargs.
            actualt = get_proper_type(actual_arg_type(ai))

            # Special case for union of equal sized tuples.
            if (
                isinstance(actualt, UnionType)
                and actualt.items
                and is_equal_sized_tuples(
                    proper_types := [get_proper_type(t) for t in actualt.items]
                )
            ):
                # pick an arbitrary member
                actualt = proper_types[0]
            if isinstance(actualt, TupleType):
                # A tuple actual maps to a fixed number of formals.
                for _ in range(len(actualt.items)):
                    if fi < nformals:
                        if formal_kinds[fi] != nodes.ARG_STAR2:
                            formal_to_actual[fi].append(ai)
                        else:
                            break
                        if formal_kinds[fi] != nodes.ARG_STAR:
                            fi += 1
            else:
                # Assume that it is an iterable (if it isn't, there will be
                # an error later).
                while fi < nformals:
                    if formal_kinds[fi].is_named(star=True):
                        break
                    else:
                        formal_to_actual[fi].append(ai)
                    if formal_kinds[fi] == nodes.ARG_STAR:
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
        allow_unpack: bool = False,
    ) -> Type:
        """Return the actual (caller) type(s) of a formal argument with the given kinds.

        If the actual argument is a tuple *args, return the next individual tuple item that
        maps to the formal arg.

        If the actual argument is a TypedDict **kwargs, return the next matching typed dict
        value type based on formal argument name and kind.

        This is supposed to be called for each formal, in order. Call multiple times per
        formal if multiple actuals map to a formal.
        """
        original_actual = actual_type
        actual_type = get_proper_type(actual_type)
        if actual_kind == nodes.ARG_STAR:
            # parse *args as one of the following:
            #    IterableType | TupleType | ParamSpecType | AnyType
            star_args_type = self.parse_star_args_type(actual_type)

            if self.context.is_tuple_instance_type(star_args_type):
                return star_args_type.args[0]
            elif isinstance(star_args_type, TupleType):
                # Get the next tuple item of a tuple *arg.
                if self.tuple_index >= len(star_args_type.items):
                    # Exhausted a tuple -- continue to the next *args.
                    self.tuple_index = 1
                else:
                    self.tuple_index += 1
                item = star_args_type.items[self.tuple_index - 1]
                if isinstance(item, UnpackType) and not allow_unpack:
                    # An unpack item that doesn't have special handling, use upper bound as above.
                    unpacked = get_proper_type(item.type)
                    if isinstance(unpacked, TypeVarTupleType):
                        fallback = get_proper_type(unpacked.upper_bound)
                    else:
                        fallback = unpacked
                    assert (
                        isinstance(fallback, Instance)
                        and fallback.type.fullname == "builtins.tuple"
                    )
                    item = fallback.args[0]
                return item
            elif isinstance(star_args_type, ParamSpecType):
                # ParamSpec is valid in *args but it can't be unpacked.
                return star_args_type
            else:
                return AnyType(TypeOfAny.from_error)
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

    def _solve_as_iterable(self, typ: Type) -> IterableType | AnyType:
        r"""Use the solver to cast a type as Iterable[T].

        Returns `AnyType` if solving fails.
        """
        from mypy.constraints import infer_constraints_for_callable
        from mypy.nodes import ARG_POS
        from mypy.solve import solve_constraints

        # We first create an upcast function:
        #    def [T] (Iterable[T]) -> Iterable[T]: ...
        # and then solve for T, given the input type as the argument.
        T = TypeVarType(
            "T",
            "T",
            TypeVarId(-1),
            values=[],
            upper_bound=AnyType(TypeOfAny.from_omitted_generics),
            default=AnyType(TypeOfAny.from_omitted_generics),
        )
        target = self.context.make_iterable_instance_type(T)
        upcast_callable = CallableType(
            variables=[T],
            arg_types=[target],
            arg_kinds=[ARG_POS],
            arg_names=[None],
            ret_type=target,
            fallback=self.context.function_type,
        )
        constraints = infer_constraints_for_callable(
            upcast_callable, [typ], [ARG_POS], [None], [[0]], self.context
        )

        (sol,), _ = solve_constraints([T], constraints)

        if sol is None:  # solving failed, return AnyType fallback
            return AnyType(TypeOfAny.from_error)
        return self.context.make_iterable_instance_type(sol)

    def as_iterable_type(self, typ: Type) -> IterableType | AnyType:
        """Reinterpret a type as Iterable[T], or return AnyType if not possible.

        This function specially handles certain types like UnionType, TupleType, and UnpackType.
        Otherwise, the upcasting is performed using the solver.
        """
        p_t = get_proper_type(typ)
        if self.context.is_iterable_instance_type(p_t) or isinstance(p_t, AnyType):
            return p_t
        elif isinstance(p_t, UnionType):
            # If the type is a union, map each item to the iterable supertype.
            # the return the combined iterable type Iterable[A] | Iterable[B] -> Iterable[A | B]
            converted_types = [self.as_iterable_type(get_proper_type(item)) for item in p_t.items]

            if any(not self.context.is_iterable_instance_type(it) for it in converted_types):
                # if any item could not be interpreted as Iterable[T], we return AnyType
                return AnyType(TypeOfAny.from_error)
            else:
                # all items are iterable, return Iterable[T₁ | T₂ | ... | Tₙ]
                iterable_types = cast("list[IterableType]", converted_types)
                arg = make_simplified_union([it.args[0] for it in iterable_types])
                return self.context.make_iterable_instance_type(arg)
        elif isinstance(p_t, TupleType):
            # maps tuple[A, B, C] -> Iterable[A | B | C]
            # note: proper_elements may contain UnpackType, for instance with
            #   tuple[None, *tuple[None, ...]]..
            proper_elements = [get_proper_type(t) for t in flatten_nested_tuples(p_t.items)]
            args: list[Type] = []
            for p_e in proper_elements:
                if isinstance(p_e, UnpackType):
                    r = self.as_iterable_type(p_e)
                    if self.context.is_iterable_instance_type(r):
                        args.append(r.args[0])
                    else:
                        # this *should* never happen, since UnpackType should
                        # only contain TypeVarTuple or a variable length tuple.
                        # However, we could get an `AnyType(TypeOfAny.from_error)`
                        # if for some reason the solver was triggered and failed.
                        args.append(r)
                else:
                    args.append(p_e)
            return self.context.make_iterable_instance_type(make_simplified_union(args))
        elif isinstance(p_t, UnpackType):
            return self.as_iterable_type(p_t.type)
        elif isinstance(p_t, (TypeVarType, TypeVarTupleType)):
            return self.as_iterable_type(p_t.upper_bound)
        elif self.context.is_iterable(p_t):
            # TODO: add a 'fast path' (needs measurement) that uses the map_instance_to_supertype
            #   mechanism? (Only if it works: gh-19662)
            return self._solve_as_iterable(p_t)
        return AnyType(TypeOfAny.from_error)

    def parse_star_args_type(
        self, typ: Type
    ) -> TupleType | TupleInstanceType | ParamSpecType | AnyType:
        """Parse the type of a ``*args`` argument.

        Returns one of TupleType, TupleInstanceType, ParamSpecType or AnyType.
        Returns AnyType(TypeOfAny.from_error) if the type cannot be parsed or is invalid.
        """
        p_t = get_proper_type(typ)
        if isinstance(p_t, (TupleType, ParamSpecType, AnyType)):
            # just return the type as-is
            return p_t
        elif isinstance(p_t, TypeVarTupleType):
            return self.parse_star_args_type(p_t.upper_bound)
        elif isinstance(p_t, UnionType):
            proper_items = [get_proper_type(t) for t in p_t.items]
            # consider 2 cases:

            # 1. Union of tuple
            if all(isinstance(t, TupleType) for t in proper_items):
                proper_items = cast("list[TupleType]", proper_items)
                return self.combine_tuple_types(proper_items)
            # 2. Union of iterable types, e.g. Iterable[A] | Iterable[B]
            #    In this case return tuple[A | B, ...]
            else:
                converted_types = [self.as_iterable_type(p_i) for p_i in proper_items]
                if all(self.context.is_iterable_instance_type(it) for it in converted_types):
                    # all items are iterable, return tuple[T1 | T2 | ... | Tn, ...]
                    iterables = cast("list[IterableType]", converted_types)
                    arg = make_simplified_union([it.args[0] for it in iterables])
                    return self.context.make_tuple_instance_type(arg)
                else:
                    # some items in the union are not iterable, return AnyType
                    return AnyType(TypeOfAny.from_error)
        else:
            parsed = self.as_iterable_type(p_t)
            if self.context.is_iterable_instance_type(parsed):
                return self.context.make_tuple_instance_type(parsed.args[0])
            return AnyType(TypeOfAny.from_error)

    def process_tuple_type(self, typ: TupleType) -> tuple[TupleType, TupleType, TupleType]:
        r"""Split a tuple into 3 parts: head part, body part and tail part.

        1. A head part which is the longest finite prefix of the tuple
        2. A body part which covers all items from the first variable item to the last variable item
        3. A tail part which is the longest finite suffix of the remaining tuple

        Examples:
            - tuple[int, str] -> (tuple[int, str], tuple[()], tuple[()])
            - tuple[int, *tuple[int, ...]] -> (tuple[int], tuple[*tuple[int, ...]], tuple[()])
            - tuple[*tuple[int, ...], int] -> (tuple[())], tuple[*tuple[int, ...]], tuple[int])
            - tuple[int, *tuple[int, ...], int] -> (tuple[int], tuple[*tuple[int, ...]], tuple[int])
            - tuple[int, *tuple[int, ...], str, *tuple[str, ...], int]
              -> (head=tuple[int], variable=tuple[*tuple[int, ...], str, *tuple[str, ...]], tail=tuple[int])
        """
        head_items: list[Type] = []
        tail_items: list[Type] = []
        body_items: list[Type] = []

        flattened_items = flatten_nested_tuples(typ.items)
        generator = iter(flattened_items)

        # determine the head part
        for item in generator:
            p_t = get_proper_type(item)
            if isinstance(p_t, UnpackType):
                body_items.append(item)
                break
            head_items.append(item)
        # determine the body part
        for item in generator:
            p_t = get_proper_type(item)
            if isinstance(p_t, UnpackType):
                body_items.extend(tail_items)
                body_items.append(item)
                tail_items.clear()
            else:
                tail_items.append(item)

        # construct the return
        fallback = self.context.make_tuple_instance_type(AnyType(TypeOfAny.from_error))
        head = TupleType(head_items, fallback=fallback)
        body = TupleType(body_items, fallback=fallback)
        tail = TupleType(tail_items, fallback=fallback)

        return head, body, tail

    def combine_tuple_types(self, types: Sequence[TupleType]) -> TupleType:
        """Combine multiple tuple types into a single tuple type.

        This creates an upper bound for the union of the input tuple types.
        If all input tuples are of the same (real) size, so is the result.

        Examples:
            tuple[int, int], tuple[None, None]
                -> tuple[int | None, int | None]

            tuple[int, *tuple[int, ...], int],
            tuple[None, *tuple[None, ...], None]
                -> tuple[int | None, *tuple[int | None, ...], int | None]

            tuple[int, *tuple[int, ...], str, *tuple[str, ...], int]
            tuple[None, *tuple[None, ...], None]
                -> tuple[int | None, *tuple[int | str | None, ...], int | None]
        """
        heads: list[TupleType]
        bodies: list[TupleType]
        tails: list[TupleType]
        heads, bodies, tails = zip(*(self.process_tuple_type(typ) for typ in types))

        head_items: list[Type] = []
        body_items: list[Type] = []
        tail_items: list[Type] = []

        # 1. process all heads in parallel, stopping when one of the heads is exhausted
        head_length = min(len(head.items) for head in heads)
        for items in zip(*(head.items[:head_length] for head in heads)):
            # append the union of the items to the head part
            head_items.append(make_simplified_union(items))
        # collect all the remaining head items from generators that were not exhausted
        remaining_head_items = [item for head in heads for item in head.items[head_length:]]

        # 2. process all tails in parallel, stopping when one of the tails is exhausted
        tail_length = min(len(tail.items) for tail in tails)
        for items in zip(*(tail.items[:tail_length] for tail in tails)):
            # append the union of the items to the tail part
            tail_items.append(make_simplified_union(items))
        # collect all the remaining tail items from generators that were not exhausted
        remaining_tail_items = [item for tail in tails for item in tail.items[tail_length:]]

        # 3. process all bodies, coercing them into iterable types
        for body in bodies:
            if not body.items:
                continue
            parsed_body_type = self.as_iterable_type(body)
            if isinstance(parsed_body_type, AnyType):
                body_items.append(parsed_body_type)
            else:  # Iterable[T]
                body_items.append(parsed_body_type.args[0])
        combined_items = remaining_head_items + body_items + remaining_tail_items

        if combined_items:
            variable_type = make_simplified_union(combined_items)
            variable_part = self.context.make_tuple_instance_type(variable_type)
            return TupleType(
                [*head_items, UnpackType(variable_part), *tail_items],
                fallback=self.context.tuple_type,
            )
        return TupleType([*head_items, *body_items], fallback=self.context.tuple_type)

    def combine_finite_tuple_types(self, types: Sequence[FiniteTupleType]) -> TupleType:
        """Combine multiple finite tuple types into a single tuple type.

        The result is an upper bound for the union of the input tuple types.
        If all input tuples are of the same size, so is the result.
        If any input tuples have different sizes, the result will be a variable-length tuple.

        Note:
            We treat each input tuple as consisting of two parts:

            1. The head part, which are the first n items of the tuple, where n is the minimal length
            2. The tail part which are the remaining items of the tuple, if any.

            The resulting tuple type will be of the form

            tuple[T1, T2, ..., Tn, *tuple[U, ...]]

            where each item `Tk` of the head part is the union of the item types of the input tuples,
            and `U` is the union of all the item types of all the tail parts of each input tuple.

        Examples:
            tuple[int, int], tuple[None, None]           -> tuple[int | None, int | None]
            tuple[str], tuple[str, int]                  -> tuple[str, *tuple[int, ...]]
            tuple[str], tuple[str, str], tuple[int, int] -> tuple[str | int, *tuple[str | int, ...]]
        """
        if not types:
            raise ValueError("Expected at least one type, got empty sequence")

        # 1. get the flattened elements of each tuple
        lengths = [get_real_tuple_length(tup) for tup in types]
        flattened_tuples: list[list[Type]] = [flatten_nested_tuples(typ.items) for typ in types]
        assert all(
            len(flat_items) == length for length, flat_items in zip(lengths, flattened_tuples)
        )
        # 2. compute the lengths
        minimal_length = min(lengths)
        maximal_length = max(lengths)

        # 3. compute the head part by union-ing element-wise
        head_types = [
            make_simplified_union([tuple_items[i] for tuple_items in flattened_tuples])
            for i in range(minimal_length)
        ]

        # if no tail part return early
        if minimal_length == maximal_length:
            fallback = self.context.make_tuple_instance_type(AnyType(TypeOfAny.unannotated))
            head_type = TupleType(head_types, fallback=fallback)
            return head_type

        # 4. compute the tail part by union-ing all the remaining elements
        tail_type = make_simplified_union(
            [
                UnionType.make_union([*tuple_items[minimal_length:]])
                for tuple_items in flattened_tuples
            ]
        )
        # create the tail tuple type
        tail_tuple = self.context.make_tuple_instance_type(tail_type)

        # return head part + tail part
        fallback = self.context.make_tuple_instance_type(
            make_simplified_union([*head_types, tail_type])
        )
        result = TupleType([*head_types, UnpackType(tail_tuple)], fallback=fallback)
        return result


def normalize_finite_tuple_type(typ: FiniteTupleType) -> FiniteTupleType:
    """Normalize a tuple type to a FiniteTupleType."""
    return FiniteTupleType(flatten_nested_tuples(typ), fallback=typ.partial_fallback)


def is_tuple_instance_type(typ: Type) -> TypeIs[TupleInstanceType]:
    """Check if the type is an instance of `tuple[T, ...]`."""
    p_t = get_proper_type(typ)
    return isinstance(p_t, Instance) and p_t.type.fullname == "builtins.tuple"


def is_tuple_like_type(typ: Type) -> TypeIs[TupleLikeType]:
    """Check if the type is a tuple-like type."""
    p_t = get_proper_type(typ)
    return isinstance(p_t, TupleType | TypeVarTupleType) or is_tuple_instance_type(p_t)


def is_finite_tuple_type(typ: Type) -> TypeIs[FiniteTupleType]:
    """Check if the type is a finite tuple type."""
    p_t = get_proper_type(typ)

    if not isinstance(p_t, TupleType):
        return False
    return get_real_tuple_length(p_t) >= 0


def get_real_tuple_length(typ: TupleLikeType) -> int:
    r"""Get the length of a tuple type, or -1 if it is unbounded."""
    p_t = get_proper_type(typ)
    if isinstance(p_t, TypeVarTupleType):
        # use the upper bound of the TypeVarTuple to determine the length
        return get_real_tuple_length(p_t.upper_bound)
    if is_tuple_instance_type(p_t):
        # tuple[T, ...] is always unbounded, so return -1
        return -1
    if isinstance(p_t, TupleType):
        size = 0
        for item in p_t.items:
            proper_item = get_proper_type(item)
            if isinstance(proper_item, UnpackType):
                unpacked = get_proper_type(proper_item.type)
                assert is_tuple_like_type(unpacked)
                result = get_real_tuple_length(unpacked)
                if result == -1:
                    return -1
                size += result
            else:
                size += 1
        return size
    raise TypeError(f"Unexpected type for tuple length: {typ}")


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


def is_equal_sized_tuples(types: Sequence[ProperType]) -> TypeGuard[Sequence[TupleType]]:
    """Check if all types are tuples of the same size.

    We use `flatten_nested_tuples` to deal with nested tuples.
    Note that the result may still contain
    """
    if not types:
        return True

    iterator = iter(types)
    typ = next(iterator)
    if not isinstance(typ, TupleType):
        return False
    flattened_elements = flatten_nested_tuples(typ.items)
    if any(
        isinstance(get_proper_type(member), (UnpackType, TypeVarTupleType))
        for member in flattened_elements
    ):
        # this can happen e.g. with tuple[int, *tuple[int, ...], int]
        return False
    size = len(flattened_elements)

    for typ in iterator:
        if not isinstance(typ, TupleType):
            return False
        flattened_elements = flatten_nested_tuples(typ.items)
        if len(flattened_elements) != size or any(
            isinstance(get_proper_type(member), (UnpackType, TypeVarTupleType))
            for member in flattened_elements
        ):
            # this can happen e.g. with tuple[int, *tuple[int, ...], int]
            return False
    return True
