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
from mypy.nodes import ARG_POS, ArgKind, TypeInfo
from mypy.solve import solve_constraints
from mypy.tuple_normal_form import TupleInstanceType, TupleNormalForm
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
    flatten_nested_tuples,
    get_proper_type,
)

IterableType = NewType("IterableType", Instance)
"""Represents an instance of `Iterable[T]`."""


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
    tuple_typeinfo: TypeInfo

    @property
    def fallback_tuple(self) -> Instance:
        r"""Canonical fallback tuple type tuple[Any, ...]."""
        # NOTE: This must use ``TypeOfAny.special_form`` and not ``TypeOfAny.from_omitted_generics``,
        #   otherwise this leads to errors in dmypy SuggestionEngine.
        return Instance(self.tuple_typeinfo, [AnyType(TypeOfAny.special_form)])

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
        return isinstance(p_t, Instance) and p_t.type == self.tuple_typeinfo

    def make_tuple_instance_type(self, arg: Type) -> TupleInstanceType:
        """Create a TupleInstance type with the given argument type."""
        value = Instance(self.tuple_typeinfo, [arg])
        return cast(TupleInstanceType, value)

    def make_iterable_instance_type(self, arg: Type) -> IterableType:
        value = Instance(self.iterable_type.type, [arg])
        return cast(IterableType, value)

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
        elif isinstance(p_t, TypeVarType):
            # for a regular TypeVar, check the upper bound.
            return self.as_iterable_type(p_t.upper_bound)
        elif isinstance(p_t, TypeVarTupleType):
            # TVT -> tuple[T₁, T₂, ..., Tₙ]
            # since this is always iterable, but the variables are not known,
            # we return Iterable[Any]
            error_type = AnyType(TypeOfAny.from_error)
            return self.make_iterable_instance_type(error_type)
        elif self.is_iterable(p_t):
            # TODO: add a 'fast path' (needs measurement) that uses the map_instance_to_supertype
            #   mechanism? (Only if it works: gh-19662)
            return self._solve_as_iterable(p_t)

        # failure case, return AnyType
        return AnyType(TypeOfAny.from_error)

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
