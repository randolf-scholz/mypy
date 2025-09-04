"""Utilities for type argument inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple, NewType, cast
from typing_extensions import TypeIs

from mypy.constraints import (
    SUBTYPE_OF,
    SUPERTYPE_OF,
    infer_constraints,
    infer_constraints_for_callable,
)
from mypy.nodes import ARG_POS, ArgKind
from mypy.solve import solve_constraints
from mypy.tuple_normal_form import TupleNormalForm
from mypy.typeops import make_simplified_union
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    TupleType,
    Type,
    TypeOfAny,
    TypeVarId,
    TypeVarLikeType,
    TypeVarTupleType,
    TypeVarType,
    UnionType,
    UnpackType,
    find_unpack_in_list,
    flatten_nested_tuples,
    get_proper_type,
)

IterableType = NewType("IterableType", Instance)
"""Represents an instance of `Iterable[T]`."""
TupleInstanceType = NewType("TupleInstanceType", Instance)
"""Represents an instance of `tuple[T, ...]`."""


class ArgumentInferContext(NamedTuple):
    """Type argument inference context.

    We need this because we pass around ``Mapping`` and ``Iterable`` types.
    These types are only known by ``TypeChecker`` itself.
    It is required for ``*`` and ``**`` argument inference.

    https://github.com/python/mypy/issues/11144
    """

    mapping_type: Instance
    iterable_type: Instance
    function_type: Instance
    tuple_type: Instance

    def is_iterable(self, typ: Type) -> bool:
        """Check if the type is an iterable, i.e. implements the Iterable Protocol."""
        from mypy.subtypes import is_subtype

        return is_subtype(typ, self.iterable_type)

    def is_iterable_instance_type(self, typ: Type) -> TypeIs[IterableType]:
        """Check if the type is an Iterable[T]."""
        p_t = get_proper_type(typ)
        return isinstance(p_t, Instance) and p_t.type == self.iterable_type.type

    def is_tuple_instance_type(self, typ: Type) -> TypeIs[TupleInstanceType]:
        """Check if the type is a tuple instance, i.e. tuple[T, ...]."""
        p_t = get_proper_type(typ)
        return isinstance(p_t, Instance) and p_t.type == self.tuple_type.type

    def make_iterable_instance_type(self, arg: Type) -> IterableType:
        value = Instance(self.iterable_type.type, [arg])
        return cast(IterableType, value)

    def make_tuple_instance_type(self, arg: Type) -> TupleInstanceType:
        """Create a TupleInstance type with the given argument type."""
        value = Instance(self.tuple_type.type, [arg])
        return cast(TupleInstanceType, value)

    def as_iterable_type(self, typ: Type) -> IterableType | AnyType:
        r"""Reinterpret a type as Iterable[T], or return AnyType if not possible.

        This function specially handles certain types like UnionType, TupleType, and UnpackType.
        Otherwise, the upcasting is performed using the solver.
        """
        p_t = get_proper_type(typ)
        if self.is_iterable_instance_type(p_t) or isinstance(p_t, AnyType):
            return p_t
        elif isinstance(p_t, UnionType):
            # If the type is a union, map each item to the iterable supertype.
            # the return the combined iterable type Iterable[A] | Iterable[B] -> Iterable[A | B]
            converted_types = [self.as_iterable_type(get_proper_type(item)) for item in p_t.items]

            if any(not self.is_iterable_instance_type(it) for it in converted_types):
                # if any item could not be interpreted as Iterable[T], we return AnyType
                return AnyType(TypeOfAny.from_error)
            else:
                # all items are iterable, return Iterable[T₁ | T₂ | ... | Tₙ]
                iterable_types = cast("list[IterableType]", converted_types)
                arg = make_simplified_union([it.args[0] for it in iterable_types])
                return self.make_iterable_instance_type(arg)
        elif isinstance(p_t, TupleType):
            # maps tuple[A, B, C] -> Iterable[A | B | C]
            # note: proper_elements may contain UnpackType, for instance with
            #   tuple[None, *tuple[None, ...]]..
            proper_elements = [get_proper_type(t) for t in flatten_nested_tuples(p_t.items)]
            args: list[Type] = []
            for p_e in proper_elements:
                if isinstance(p_e, UnpackType):
                    r = self.as_iterable_type(p_e)
                    if self.is_iterable_instance_type(r):
                        args.append(r.args[0])
                    else:
                        # this *should* never happen, since UnpackType should
                        # only contain TypeVarTuple or a variable length tuple.
                        # However, we could get an `AnyType(TypeOfAny.from_error)`
                        # if for some reason the solver was triggered and failed.
                        args.append(r)
                else:
                    args.append(p_e)
            return self.make_iterable_instance_type(make_simplified_union(args))
        elif isinstance(p_t, UnpackType):
            return self.as_iterable_type(p_t.type)
        elif isinstance(p_t, (TypeVarType, TypeVarTupleType)):
            return self.as_iterable_type(p_t.upper_bound)
        elif self.is_iterable(p_t):
            # TODO: add a 'fast path' (needs measurement) that uses the map_instance_to_supertype
            #   mechanism? (Only if it works: gh-19662)
            return self._solve_as_iterable(p_t)
        # return AnyType(TypeOfAny.from_error)
        error_type = AnyType(TypeOfAny.from_error)
        return self.make_iterable_instance_type(error_type)

    def _solve_as_iterable(self, typ: Type, /) -> IterableType | AnyType:
        r"""Use the solver to cast a type as Iterable[T].

        Returns `AnyType` if solving fails.
        """
        from mypy.constraints import infer_constraints_for_callable
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
        target = self.make_iterable_instance_type(T)
        upcast_callable = CallableType(
            variables=[T],
            arg_types=[target],
            arg_kinds=[ARG_POS],
            arg_names=[None],
            ret_type=target,
            fallback=self.function_type,
        )
        constraints = infer_constraints_for_callable(
            upcast_callable, [typ], [ARG_POS], [None], [[0]], self
        )

        (sol,), _ = solve_constraints([T], constraints)

        if sol is None:  # solving failed, return AnyType fallback
            error_type = AnyType(TypeOfAny.from_error)
            return self.make_iterable_instance_type(error_type)
        return self.make_iterable_instance_type(sol)

    def get_tuple_item(self, tup: TupleType, index: int) -> Type | None:
        r"""Get the type of indexing the tuple.

        Assuming the tuple type is in Tuple Normal Form, then the result is:

        tuple[P1, ..., Pn, *Vs, S1, ..., Sm] @ index =
            Iterable_type[Vs]  if index < -m
            S[index]           if -m ≤ index < 0
            P[index]           if 0 ≤ index < n
            Iterable_type[Vs]  if index ≥ n

        If the tuple has no variadic part, then it works just like regular indexing,
        but returns None if the index is out of bounds.
        """
        proper_items = tup.proper_items
        unpack_index = find_unpack_in_list(proper_items)

        if unpack_index is None:
            try:
                return proper_items[index]
            except IndexError:
                return None

        N = len(proper_items)
        if unpack_index - N < index < unpack_index:
            return proper_items[index]

        item = proper_items[unpack_index]
        assert isinstance(item, UnpackType)
        unpacked = get_proper_type(item.type)

        # convert to Iterable[T] and return T
        parsed_unpack = self.as_iterable_type(unpacked)
        return parsed_unpack.args[0]

    def get_expanded_tuple_slice(self, tup: TupleType, the_slice: slice) -> TupleType:
        r"""Slice a variadic tuple as if it was expanded into an infinite sequence.

        Args:
            tup: The tuple to slice.
            slice: The slice to apply. Only supports the following cases:
            - both start and stop have the same sign
            - start is None and stop is a non-negative integer
            - start is a negative integer and stop is None

        Examples:
            tuple[P1, P2, P3, *tuple[B, ...], S1, S2, S3]:
            - [:3] -> tuple[P1, P2, P3]
            - [1:4] -> tuple[P2, P3, B]
            - [-4:-1] -> tuple[B, S1, S2]
            - [-4:] -> tuple[B, S1, S2, S3]
            - [:12:2] -> tuple[P1, P3, B, B, B, B]
        """
        # TODO: use match-case when we drop Python 3.9 support
        start = the_slice.start
        stop = the_slice.stop
        step = the_slice.step

        if start is None:
            if stop is None:
                raise ValueError("slice start and stop cannot both be None")
            elif stop < 0:
                raise ValueError("if slice start is None, stop must be non-negative")
            else:
                pass
        elif stop is None:
            if start >= 0:
                raise ValueError("if slice stop is None, start must be negative")
            else:
                pass

    def get_tuple_slice(self, tup: TupleType, the_slice: slice) -> TupleType:
        r"""Get a special slice from the tuple.

        If the tuple has no variadic part, this works like regular slicing.
        However, if the tuple has a variadic part, then, depending on the indices,
        this does something special:

        1. If both start and stop are same signed integers (both non-negative or both negative),
           then we slice as if the variadic part as expanded into an infinite sequence
           of items, whose type we get by casting the unpack type to Iterable[T] and taking T.
           This supports step values other than 1 or -1.
        2. In all other cases,
        It is assumed the tuple is in Tuple Normal Form.

        t = tuple[P1, ..., Pn, *Vs, S1, ..., Sm]

        Depending on the sign of start, the starting point is determined as follows:
           If start ≥ 0, then start = min(start, n)
           If start < 0, then start = max(start, -m)
        which corresponds to taking items from the prefix if start is non-negative,
        and from the suffix if start is negative, but never going beyond the variadic part.

        slices that 'traverse' the variadic part always include the entire variadic part,
        irrespective of the step size.

        The slice is constructed as follows:
        - if both start and stop are within the prefix or both within the suffix,
          just do regular slicing
        - If they traverse the variadic part, create two slices and glue them with the variadic part in between.
        """
        proper_items = tup.proper_items
        unpack_index = find_unpack_in_list(proper_items)

        if unpack_index is None:
            return TupleType(proper_items[the_slice], self.tuple_type)

        variadic_part = proper_items[unpack_index]
        assert isinstance(variadic_part, UnpackType)
        variadic_as_iterable = self.as_iterable_type(variadic_part.type)
        iterable_type = variadic_as_iterable.args[0]

        N = len(proper_items)
        prefix_length = len(tup.prefix)
        suffix_length = len(tup.suffix)

        # get the corrected start and stop indices
        start = (
            0
            if the_slice.start is None
            else max(min(-suffix_length, -1), min(the_slice.start, prefix_length))
        )
        stop = (
            -1
            if the_slice.stop is None
            else max(min(-suffix_length, -1), min(the_slice.stop, prefix_length))
        )
        step = the_slice.step

        if start >= 0 and stop >= 0:
            items = [
                proper_items[i] if i < prefix_length else iterable_type
                for i in range(start, stop, step)
            ]
        elif start >= 0 and stop < 0:
            if step != 1:
                raise ValueError("step must be 1")
            items = [
                *proper_items[start:unpack_index:step],
                variadic_part,
                *proper_items[unpack_index + 1 : (stop % N) + 1 : step],
            ]
        elif start < 0 and stop >= 0 and step < 0:
            if step != -1:
                raise ValueError("step must be -1")
            items = [
                *proper_items[start : unpack_index + 1 : step],
                variadic_part,
                *proper_items[unpack_index - 1 : (stop % N) + 1 : step],
            ]
        elif start < 0 and stop < 0:
            items = [
                proper_items[i] if i >= -suffix_length else iterable_type
                for i in range(start, stop, step)
            ]
        else:
            # empty slice
            items = []
        return TupleType(items, self.tuple_type)

    def make_tuple_type(self, items: Sequence[Type], /) -> TupleType:
        r"""Create a proper TupleType from the given item types."""
        tnf = TupleNormalForm.from_items(items)
        return tnf.materialize(context=self)

    def concatenate_tuples(
        self,
        args: Iterable[TupleType],
        /,
        *,
        coerce_unpacks: bool = True,
        coerce_unions: bool = True,
    ) -> TupleType:
        r"""Concatenate multiple tuple types.

        Args:
            coerce_unpacks: if True, coerce multiple variadic parts into a single
                variadic part of the union of their argument types. (default: True)
            coerce_unions: if True, coerce unions in variadic parts into a single
                variadic part of the union of their argument types. (default: True)
        """
        if not coerce_unpacks or not coerce_unions:
            # https://discuss.python.org/t/should-unions-of-tuples-tvts-be-allowed-inside-unpack/102608
            raise NotImplementedError

        tnfs = [TupleNormalForm.from_tuple_type(t) for t in args]
        tnf = TupleNormalForm.combine_concat(tnfs)
        return tnf.materialize(context=self)

    def tuple_from_items(self, items: Sequence[Type]) -> TupleType:
        r"""Create a TupleType from the given item types.

        This is a thin wrapper around TupleNormalForm.from_items.
        """
        tnf = TupleNormalForm.from_items(items)
        return tnf.materialize(context=self)


def infer_function_type_arguments(
    callee_type: CallableType,
    arg_types: Sequence[Type | None],
    arg_kinds: list[ArgKind],
    arg_names: Sequence[str | None] | None,
    formal_to_actual: list[list[int]],
    context: ArgumentInferContext,
    strict: bool = True,
    allow_polymorphic: bool = False,
) -> tuple[list[Type | None], list[TypeVarLikeType]]:
    """Infer the type arguments of a generic function.

    Return an array of lower bound types for the type variables -1 (at
    index 0), -2 (at index 1), etc. A lower bound is None if a value
    could not be inferred.

    Arguments:
      callee_type: the target generic function
      arg_types: argument types at the call site (each optional; if None,
                 we are not considering this argument in the current pass)
      arg_kinds: nodes.ARG_* values for arg_types
      formal_to_actual: mapping from formal to actual variable indices
    """
    # Infer constraints.
    constraints = infer_constraints_for_callable(
        callee_type, arg_types, arg_kinds, arg_names, formal_to_actual, context
    )

    # Solve constraints.
    type_vars = callee_type.variables
    return solve_constraints(type_vars, constraints, strict, allow_polymorphic)


def infer_type_arguments(
    type_vars: Sequence[TypeVarLikeType],
    template: Type,
    actual: Type,
    is_supertype: bool = False,
    skip_unsatisfied: bool = False,
) -> list[Type | None]:
    # Like infer_function_type_arguments, but only match a single type
    # against a generic type.
    constraints = infer_constraints(template, actual, SUPERTYPE_OF if is_supertype else SUBTYPE_OF)
    return solve_constraints(type_vars, constraints, skip_unsatisfied=skip_unsatisfied)[0]
