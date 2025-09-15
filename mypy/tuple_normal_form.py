from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple, NewType, cast
from typing_extensions import TypeAlias as _TypeAlias, TypeGuard, TypeIs

from mypy.maptype import map_instance_to_supertype
from mypy.nodes import TypeInfo
from mypy.typeops import make_simplified_union
from mypy.types import (
    AnyType,
    Instance,
    ParamSpecType,
    ProperType,
    TupleType,
    Type,
    TypeList,
    TypeOfAny,
    TypeVarTupleType,
    UninhabitedType,
    UnionType,
    UnpackType,
    flatten_nested_tuples,
    get_proper_type,
)

if TYPE_CHECKING:
    from mypy.infer import ArgumentInferContext


def show(*args: object) -> None:
    if False:
        print(*args)


# TODO: Actually properly define VariadicType if ever `UnpackType` and `UnionType` are made generic.
#     type VariadicType = UnpackType[TypeList | UnionType[VariadicType]]
VariadicType = NewType("VariadicType", UnpackType)

FlatTuple = NewType("FlatTuple", TupleType)
"""A tuple type that has been flattened; any UnpackType should only contain TypeVarTupleType or TupleInstance."""
FiniteTuple = NewType("FiniteTuple", TupleType)
"""Represents an instance of `tuple[T1, T2, ..., Tn]` with a finite number of items."""
TupleLikeType: _TypeAlias = "TupleType | TypeVarTupleType | TupleInstanceType | FiniteTuple"
r"""Types that are considered tuples or tuple-like."""
AbstractUnpackType = NewType("AbstractUnpackType", UnpackType)
r"""An UnpackType that may contain unexpected members, such as TypeList or UnionType."""
TupleInstanceType = NewType("TupleInstanceType", Instance)
"""Represents an instance of `tuple[T, ...]`."""


def get_std_tuple_typeinfo(typ: TupleType, /) -> TypeInfo:
    """Extract the TypeInfo of 'builtins.tuple' from a TupleType."""
    fallback = typ.partial_fallback
    if fallback.type.fullname == "builtins.tuple":
        return fallback.type

    # this can happen when the fallback is a NamedTuple subclass
    # in this case, we look for 'builtins.tuple' in the MRO
    for base in fallback.type.mro:
        if base.fullname == "builtins.tuple":
            return base
    raise RuntimeError("Could not find builtins.tuple in the MRO of the fallback type")


class TupleHelper:
    """Helper class for certain tuple operations."""

    tuple_typeinfo: TypeInfo

    def __init__(self, tuple_type: TypeInfo | TupleType | Instance) -> None:
        if isinstance(tuple_type, Instance):
            tuple_type = tuple_type.type

        if isinstance(tuple_type, TupleType):
            tuple_type = get_std_tuple_typeinfo(tuple_type)

        if tuple_type.fullname != "builtins.tuple":
            raise ValueError(f"Expected 'builtins.tuple' TypeInfo, got {tuple_type}")
        self.tuple_typeinfo = tuple_type

    @property
    def std_tuple(self) -> Instance:
        """return tuple[Any, ...]"""
        return Instance(self.tuple_typeinfo, [AnyType(TypeOfAny.from_omitted_generics)])

    def is_tuple_instance_type(self, typ: Type, /) -> TypeIs[TupleInstanceType]:
        """Check if the type is a tuple instance, i.e. tuple[T, ...]."""
        p_t = get_proper_type(typ)
        return isinstance(p_t, Instance) and p_t.type == self.tuple_typeinfo

    def is_tuple_instance_subtype(self, typ: Type) -> bool:
        """Check if the type is a subtype of tuple[T, ...] for some T."""
        from mypy.subtypes import is_subtype

        if not isinstance(typ, Instance):
            return False
        if typ.type == self.tuple_typeinfo:
            return True
        # otherwise, check if it is a subtype of tuple[Any, ...]
        return is_subtype(typ, self.std_tuple)

    def as_tuple_instance_type(self, typ: Type, /) -> TupleInstanceType:
        r"""Upcast a subtype of tuple[T, ...] to tuple[T, ...]."""
        if not self.is_tuple_instance_subtype(typ):
            raise ValueError(f"Type {typ} is not a subtype of tuple[T, ...]")
        # TODO: does this always give the same result as the solver?
        return map_instance_to_supertype(typ, self.tuple_typeinfo)

    def make_tuple_instance_type(self, arg: Type, /) -> TupleInstanceType:
        """Create a TupleInstance type with the given argument type."""
        value = Instance(self.tuple_typeinfo, [arg])
        return cast(TupleInstanceType, value)

    def make_tuple_type(self, items: Sequence[Type], /) -> TupleType:
        r"""Create a proper TupleType from the given item types."""
        self._validate_items_for_tuple_type(items)
        # make the fallback type
        fallback = self._make_fallback_for_tuple_items(items)
        return TupleType(items, fallback=fallback)

    def _make_fallback_for_tuple_items(self, items: Sequence[Type]) -> Instance:
        item_types = []
        for item in flatten_nested_tuples(items):
            if isinstance(item, UnpackType):
                unpacked = get_proper_type(item.type)
                if self.is_tuple_instance_type(unpacked):
                    # unpacked is tuple[T, ...], return T
                    item_types.append(unpacked.args[0])
                elif isinstance(unpacked, TypeVarTupleType):
                    # unpacked is a TypeVarTuple, return Any
                    item_types.append(AnyType(TypeOfAny.from_omitted_generics))
                elif isinstance(unpacked, ParamSpecType):
                    # assert (
                    #     unpacked.flavor == ParamSpecFlavor.ARGS
                    # ), f"items={items}, {unpacked.flavor=}"
                    item_types.append(AnyType(TypeOfAny.from_omitted_generics))
                else:
                    assert False, f"Unexpected unpacked type: {unpacked}"
            else:
                item_types.append(item)

        combined_item_type = make_simplified_union(item_types)
        return self.make_tuple_instance_type(combined_item_type)

    def _validate_items_for_tuple_type(self, items: Sequence[Type]) -> None:
        """Validate that the items are valid for a TupleType."""
        seen_unpack = 0
        for item in flatten_nested_tuples(items):
            if isinstance(item, UnpackType):
                seen_unpack += 1
                unpacked = get_proper_type(item.type)
                if not (
                    self.is_tuple_instance_type(unpacked)
                    or isinstance(unpacked, (TypeVarTupleType, ParamSpecType))
                ):
                    raise ValueError(
                        f"UnpackType must contain tuple[T, ...] or TypeVarTuple, got {unpacked}"
                    )
        if seen_unpack > 1:
            raise ValueError("TupleType can only have one UnpackType")

    def _get_variadic_item_type(self, tup: TupleType, /) -> Type | None:
        """Get the type of the variadic part of a tuple, or None if there is no variadic part."""
        unpack_index = tup.unpack_index
        if unpack_index is None:
            return None

        item = tup.proper_items[unpack_index]
        assert isinstance(item, UnpackType)

        return self._get_variadic_item_type_from_unpack(item)

    def _get_variadic_item_type_from_unpack(self, unpack: UnpackType, /) -> Type:
        """Get the type of the variadic part from an UnpackType."""
        unpacked = get_proper_type(unpack.type)
        if self.is_tuple_instance_type(unpacked):
            # unpacked is tuple[T, ...], return T
            return unpacked.args[0]
        elif isinstance(unpacked, TypeVarTupleType):
            # unpacked is a TypeVarTuple, return Any
            return AnyType(TypeOfAny.from_omitted_generics)
        elif isinstance(unpacked, ParamSpecType):
            # need for some specific cases like
            return unpack
        else:
            assert False, f"Unexpected unpacked type: {unpacked}"

    def get_item(self, tup: TupleType, index: int) -> Type | None:
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
        unpack_index = tup.unpack_index

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
        return self._get_variadic_item_type_from_unpack(item)

    def get_slice(
        self, tup: TupleType, start: int | None, stop: int | None, step: int = 1
    ) -> TupleType:
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
        unpack_index = tup.unpack_index

        if unpack_index is None:
            return self.make_tuple_type(proper_items[start:stop:step])

        variadic_part = proper_items[unpack_index]
        assert isinstance(variadic_part, UnpackType)
        iterable_type = self._get_variadic_item_type_from_unpack(variadic_part)

        prefix_length = len(tup.prefix)
        suffix_length = len(tup.suffix)

        # clip start and stop to the valid range [-suffix_length, +prefix_length]
        clip = lambda x: max(min(-suffix_length, -1), min(x, prefix_length))
        start = None if start is None else clip(start)
        stop = None if stop is None else clip(stop)
        step = 1 if step is None else step

        if (start is None and start >= 0) and (stop is not None and stop >= 0):
            start = 0 if start is None else start
            items = [
                proper_items[i] if i < prefix_length else iterable_type
                for i in range(start, stop, step)
            ]
        elif (start is None or start >= 0) and (stop is None or stop < 0) and step == 1:
            items = [
                *proper_items[start:unpack_index:step],
                variadic_part,
                *proper_items[unpack_index + 1 : stop : step],
            ]
        elif (start is None or start < 0) and (stop is None or stop >= 0) and step == -1:
            items = [
                *proper_items[start : unpack_index + 1 : step],
                variadic_part,
                *proper_items[unpack_index - 1 : stop : step],
            ]
        elif (start is not None and start < 0) and (stop is None or stop < 0):
            stop = 0 if stop is None else stop
            items = [
                proper_items[i] if i >= -suffix_length else iterable_type
                for i in range(start, stop, step)
            ]
        else:
            # empty slice
            items = []

        return self.make_tuple_type(items)


def _is_empty_unpack(typ: Type, /) -> TypeGuard[UnpackType]:
    """Check if the variadic part is empty."""
    proper_arg = get_proper_type(typ)
    if not isinstance(proper_arg, UnpackType):
        return False

    content = get_proper_type(proper_arg.type)
    if isinstance(content, UninhabitedType):
        return True
    elif isinstance(content, TypeList | UnionType | TupleType):
        return all(_is_empty_unpack(item) for item in content.items)
    # fallback: TypeVarTupleType, tuple[T, ...], list[T], etc.
    # TODO: should we try converting to Iterable[T] and check if T is UninhabitedType?
    return False


def _is_non_empty_unpack(typ: Type, /) -> TypeGuard[UnpackType]:
    """Check if the variadic part is non-empty."""
    proper_arg = get_proper_type(typ)
    if not isinstance(proper_arg, UnpackType):
        return False

    content = get_proper_type(proper_arg.type)
    if isinstance(content, UninhabitedType):
        return False
    elif isinstance(content, TypeList | UnionType | TupleType):
        return any(_is_non_empty_unpack(item) for item in content.items)
    # fallback: TypeVarTupleType, tuple[T, ...], list[T], etc.
    # TODO: should we try converting to Iterable[T] and check if T is UninhabitedType?
    return True


class TupleNormalForm(NamedTuple):
    r"""For a given tuple type `t`, it's Normal Form is defined as the representation:

        t = tuple[P1, ..., Pn, *Vs?, S1, ..., Sm]

    where:

        - P1, ..., Pn is the maximal statically known finite prefix of the tuple,
        - Vs is the (potentially missing) variable part of the tuple,
        - S1, ..., Sm is the maximal statically known finite suffix of the tuple
        - If the tuple has no variable part, both Vs and S1, ..., Sm are empty.

    Note:
        Special attention must be paid when constructing a tuple from a TupleNormalForm,
        since the variadic part contains unexpected `UnpackType` members,
        specifically `TypeList`, `UnionType`, `UninhabitedType` and potentially `ParamSpecType`.

        - `UnpackType[TypeList[T1, ..., Tn]]` should be interpreted as unpacking
           a concatenation of multiple variadic parts, e.g. `f(*[T1, ..., Tn])`
        - `UnpackType[UnionType[T1, ..., Tn]]` should be interpreted as unpacking
            a union of multiple variadic items `x: T1 | T2 | ... | Tn; f(*x)`
        - Note that ``*Union[A, B] is equivalent to ``Union[*A, *B]``
        - Note that ``*TypeList[*TypeList[A1, ..., An], *TypeList[B1, ... Bm]]``
          is equivalent to ``*TypeList[A1, ..., An, B1, ..., Bm]``

    Attributes:
        - prefix: the longest statically known finite prefix of the tuple
        - variadic: an improper `UnpackType` representing the variable part of the tuple
          - This can contain things that are usually not allowed in `UnpackType`, in particular:
            - `TypeList` representing concatenation of multiple variadic parts
            - `UnionType` representing a union of multiple variadic parts
            - `ParamSpecType` representing `*P.args`
        - suffix: the longest statically known finite suffix of the remaining tuple
    """

    prefix: Sequence[ProperType]
    variadic: UnpackType
    suffix: Sequence[ProperType]

    # def __init__(self, prefix: Sequence[ProperType], variadic, suffix: Sequence[ProperType]) -> None:
    #     self.prefix = prefix
    #     self.variadic = variadic
    #     self.suffix = suffix

    @property
    def is_variadic(self) -> bool:
        """Inspect if the tuple has a variable part."""
        return not _is_empty_unpack(self.variadic)

    @property
    def minimum_length(self) -> int:
        """The minimum length of the tuple represented by this TupleNormalForm.

        If the tuple is not variadic, this coincides with the actual length.
        """
        # NOTE: Technically the variadic part could produce additional items,
        #    if multiple unpacks are present, e.g.
        #    tuple[int, *tuple[int, ...], str, *tuple[str, ...], str]
        #    is at least length 3.
        #    However we treat this as if it were tuple[int, *tuple[T, ...], str]
        return len(self.prefix) + len(self.suffix)

    @staticmethod
    def from_star_param(star_param: Type, /) -> TupleNormalForm:
        """Create a TupleNormalForm from the type of a ``*args: T`` annotation.

        During Semantic Analysis, the type of `*args: T` is not always wrapped in `UnpackType`.
        in particular, ``*args: int`` just gives `int`.

        See Also: `from_star_arg` for types passed as star arguments.
        """
        p_t = get_proper_type(star_param)
        if isinstance(p_t, UnpackType):
            # we can use the same logic as from_star_arg
            return TupleNormalForm.from_star_arg(p_t)
        elif isinstance(p_t, ParamSpecType):
            # ParamSpecType is always variadic
            variadic_part = UnpackType(p_t, from_star_syntax=True)
            return TupleNormalForm([], variadic_part, [])
        else:
            # otherwise we have an annotation like `*args: int`
            # this should be treated as if it were `*args: *tuple[int, ...]`
            # we deal with this by representing it as Unpack[<TypeList int>]
            # despite being conceptually equal to a single item, during materialization
            # this will be converted back to tuple[int, ...] in
            variadic_part = UnpackType(TypeList([p_t]), from_star_syntax=True)
            return TupleNormalForm([], variadic_part, [])

    @staticmethod
    def from_star_arg(star_arg: Type, /) -> TupleNormalForm:
        """Create a TupleNormalForm from a type that was passed as a star argument.

        Uses special cases for tuple types and unions of tuples.
        Note that during typ analysis, the types are not wrapped in `UnpackType`,
        so we should not see `UnpackType` here.

        On the flipside, when we see *any* variadic type, including
        `TypeVarTupleType`, `ParamSpec.args`, `list[T]`, etc., then we wrap it in
        an `UnpackType` when adding it to the variadic part of the TupleNormalForm.

        Examples:
            - list[int]                    -> [], [list[int]], []
            - tuple[int, str]              -> [int, str], [], []
            - tuple[*tuple[str, ...], str] -> [], [*tuple[str, ...]], [str]
            - Ts                           -> [], [*Ts], []
            - P.args                       -> [], [P], []
            - list[Never]                  -> [], [list[Never]], []

        Some special casing is applied to unions:
            - list[int] | list[str] -> [], [list[int] | list[str]], []
            - tuple[int, str] | list[str] -> [], [tuple[int, str] | list[str]], []

        """
        p_t = get_proper_type(star_arg)
        if isinstance(p_t, UnpackType):
            # Note: mypy is inconsistent regarding wrapping types in UnpackType.
            # def foo(*args: *tuple[int, ...]): ...;
            # def outer(*args: *tuple[int, ...]):
            #     foo(*x)    # x --> Instance(tuple), not UnpackType
            # def bar(*args: *Ts): ...;
            # def outer(*args: *Ts): ...
            #     bar(*args)  # args --> UnpackType(TypeVarTupleType)
            p_t = get_proper_type(p_t.type)

        assert not isinstance(p_t, UnpackType), f"Unexpected UnpackType: {star_arg}"

        # special case single tuple
        if isinstance(p_t, TupleType):
            return TupleNormalForm.from_tuple_type(p_t)

        # special case union of tuples
        elif isinstance(p_t, UnionType):
            # if all items are tuples, we can split them
            tnfs = [TupleNormalForm.from_star_arg(x) for x in p_t.proper_items]
            return TupleNormalForm.combine_union(tnfs)

        # assume that the star args is some variadic type,
        #    e.g. ParamSpec, TypeVarTupleType, tuple[T, ...], list[T], etc.
        # wrap it in UnpackType[TypeList].
        else:
            variadic_part = UnpackType(star_arg, from_star_syntax=True)
            return TupleNormalForm([], variadic_part, [])

    @staticmethod
    def from_tuple_type(typ: TupleType, /) -> TupleNormalForm:
        r"""Split a tuple into 3 parts: head part, body part and tail part.

        1. A head part which is the longest finite prefix of the tuple
        2. A body part which covers all items from the first variable item to the last variable item
        3. A tail part which is the longest finite suffix of the remaining tuple

        If the body part is empty, the tail part is empty as well.
        The body part, if non-empty, always starts and ends with a variable item (UnpackType).
        Note that according to the current specification, the body part may contain at maximum
        a single variable item (UnpackType), so the body part actually should at maximum be
        of length 1. This implementation should still work if that specification changes in the future.

        Examples:
            - tuple[int, str] -> ([int, str], [], [])
            - tuple[int, *tuple[int, ...]] -> ([int], [*tuple[int, ...]], [])
            - tuple[*tuple[int, ...], int] -> ([], [*tuple[int, ...]], [int])
            - tuple[int, *tuple[int, ...], int] -> ([int], [*tuple[int, ...]], [int])
            - tuple[int, *tuple[int, ...], str, *tuple[str, ...], int]
              -> ([int], [*tuple[int, ...], str, *tuple[str, ...]], [int])
        """
        return TupleNormalForm.from_items(typ.items)

    @staticmethod
    def from_items(items: Iterable[Type], /) -> TupleNormalForm:
        head_items: list[ProperType] = []
        tail_items: list[ProperType] = []
        body_items: list[ProperType] = []
        seen_variadic = False

        # determine the head, body and tail parts
        for item in flatten_nested_tuples(items):
            if _is_empty_unpack(item):
                # skip empty unpacks
                continue
            elif _is_non_empty_unpack(item):
                seen_variadic = True
                body_items.extend(tail_items)
                body_items.append(item)
                tail_items.clear()
            elif seen_variadic:
                tail_items.append(item)
            else:
                head_items.append(item)

        # the variadic part is the unpacking of the concatenation of all body items
        # formally represented by a UnpackType[TypeList[...]]
        body = UnpackType(TypeList(body_items), from_star_syntax=True)
        return TupleNormalForm(head_items, body, tail_items)

    @staticmethod
    def combine_union(args: Sequence[TupleNormalForm], /) -> TupleNormalForm:
        """Combine a union of TupleNormalForm into a single TupleNormalForm.

        - The head will be the element-wise union of all heads, stopping when one of the heads is exhausted.
        - the body will be a special UnionType[TypeList[...]] construct
        - The tail will be the element-wise union of all tails, stopping when one of the tails is exhausted.
          Note that for body-less union members, any head items that were not consumed when creating the
          joint head are prepended to the tail.

        In particular, if any single one of the inputs is head-less, then the resulting head is also empty.

        Examples:
            tuple[int, int], tuple[None, None]
                --> [int | None], *TypeList[], [int | None]

            tuple[int, *tuple[int, ...], int],
            tuple[None, *tuple[None, ...], None]
                --> [int | None],
                    *Union[
                        TypeList[*tuple[int, ...]],
                        TypeList[*tuple[None, ...]]
                    ],
                    Unpack[*tuple[int, ...] | *tuple[None, ...]]
                    [int | None]

            tuple[int, *tuple[int, ...], str, *tuple[str, ...], int]
            tuple[None, *tuple[None, ...], None]
                -->  [int | None],
                    *Union[
                        TypeList[*tuple[int, ...], str, *tuple[str, ...]],
                        TypeList[*tuple[None, ...]],
                    ],
                    [int | None]

            tuple[int, str], list[None]
                --> [],
                    *Union[
                        TypeList[int, str],
                        TypeList[*list[None]]
                    ],
                    []

            tuple[int, int] | tuple[*tuple[int, ...], int]
                --> [], [[[int], [*tuple[int, ...]]], [int]
        """
        heads: list[list[ProperType]]
        bodies: list[UnpackType]
        tails: list[list[ProperType]]
        remaining_head_items: list[list[ProperType]]
        remaining_tail_items: list[list[ProperType]]
        remaining_body_items: list[list[ProperType]]
        target_head_items: list[ProperType] = []
        target_tail_items: list[ProperType] = []

        # split each tuple
        heads, bodies, tails = zip(*args)

        # 1. process all heads in parallel, stopping when one of the heads is exhausted
        shared_head_length = min(len(head) for head in heads)
        for items in zip(*(head[:shared_head_length] for head in heads)):
            # append the union of the items to the head part
            target_head_items.append(make_simplified_union(items))
        # collect all the remaining head items from generators that were not exhausted
        remaining_head_items = [head[shared_head_length:] for head in heads]

        # If a tuple has no body items, prepend the remaining head items to the tail.
        # This addresses cases like combining `tuple[A, B, C]` with `tuple[X, *tuple[Y, ...], Z]`.
        # which should yield tuple[A | X, *tuple[B | Y, ...], C | Z]
        for remaining_head, body, tail in zip(remaining_head_items, bodies, tails):
            if _is_empty_unpack(body):
                # move all remaining head items to the start of the tail
                _tail = tail[:]
                tail.clear()
                tail.extend(remaining_head)
                tail.extend(_tail)
                remaining_head.clear()

        # 2. process all tails in parallel, in reverse, stopping when one of the tails is exhausted
        shared_tail_length = min(len(tail) for tail in tails)
        for items in zip(*(tail[-1 : -shared_tail_length - 1 : -1] for tail in tails)):
            # append the union of the items to the tail part
            target_tail_items.append(make_simplified_union(items))
        # collect all the remaining tail items from generators that were not exhausted
        target_tail_items.reverse()  # reverse to maintain original order
        remaining_tail_items = [tail[: len(tail) - shared_tail_length] for tail in tails]
        # note: do not use tail[:-shared_tail_length]; breaks when shared_tail_length=0

        # 3. process all bodies
        assert len(remaining_head_items) == len(remaining_tail_items) == len(bodies)
        remaining_body_items = [
            [
                *remaining_head,
                body,
                *remaining_tail,
            ]  # TODO: expand body in case like Unpack[<TypeList >] ?
            for remaining_head, body, remaining_tail in zip(
                remaining_head_items, bodies, remaining_tail_items
            )
        ]

        # 4. collected all items that will be put into the variable part
        # Note: if the collection is empty, this will give UninhabitedType.
        joined_bodies = UnionType.make_union(
            [UnpackType(TypeList(body_items)) for body_items in remaining_body_items]
        )

        # 5. combine all parts into a TupleNormalForm
        return TupleNormalForm(target_head_items, UnpackType(joined_bodies), target_tail_items)

    @staticmethod
    def combine_concat(tnfs: Sequence[TupleNormalForm]) -> TupleNormalForm:
        """Combine sequence of TupleNormalForm into a single TupleNormalForm.

        essentially converts ``(*x1, ..., *xn)`` -> ``*x` where x = [*x1, ..., *xn]``
        """
        if len(tnfs) == 0:
            return TupleNormalForm([], UnpackType(UninhabitedType()), [])

        if len(tnfs) == 1:
            return tnfs[0]

        items = (
            item
            for item in chain.from_iterable(
                (*tnf.prefix, tnf.variadic, *tnf.suffix) for tnf in tnfs
            )
        )
        return TupleNormalForm.from_items(items)

    def materialize(self, context: ArgumentInferContext) -> TupleType:
        """Construct the actual TupleType from the TupleNormalForm.

        Since this method needs access to the `TypeInfo` of `builtins.tuple`
        and `typing.Iterable`, we require the caller to provide an `ArgumentInferContext`.
        """
        helper = _TupleConstructor(context)
        result = helper.make_tuple_type(self)
        show(f"Materialized TupleNormalForm\n\t{self}\n\t{result}")
        return result


class _TupleConstructor:
    """Helper class responsible for constructing an actual TupleType from a TupleNormalForm."""

    def __init__(self, context: ArgumentInferContext) -> None:
        self.context = context

    def make_tuple_type(self, tnf: TupleNormalForm) -> TupleType:
        r"""Construct an actual TupleType from the TupleNormalForm.

        Combines all members of the variadic part into a single tuple[T, ...] type.
        This creates an upper bound for the original `star_args` argument.

        Pays special attention to the variadic part, which may contain unexpected
        `UnpackType` members, namely `UnionType[TypeList]`.
        """

        # parse the variadic part. UninhabitedType indicated no variadic part.
        # AnyType indicates we could not properly parse the variadic part.
        parsed_variadic_part = self.parse_variadic_type(tnf.variadic)
        show(tnf, parsed_variadic_part)

        # check whether the unpack is considered empty
        unpacked = get_proper_type(parsed_variadic_part.type)
        is_empty_unpack = isinstance(unpacked, UninhabitedType)

        # if isinstance(unpacked, (TypeVarTupleType, ParamSpecType)):
        #     is_empty = False
        # elif isinstance(unpacked, TupleType):
        #     is_empty = not unpacked.proper_items
        # elif self.context.is_tuple_instance_type(unpacked):
        #     # we treat *tuple[Never, ...] as empty
        #     is_empty = isinstance(unpacked.args[0], UninhabitedType)
        # else:
        #     raise TypeError(f"unexpected type {unpacked!r}")

        if is_empty_unpack:
            assert not tnf.suffix, f"Failed to correctly parse TupleNormalForm: {tnf}"
            return TupleType([*tnf.prefix], fallback=self.context.fallback_tuple)

        return TupleType(
            [*tnf.prefix, parsed_variadic_part, *tnf.suffix], fallback=self.context.fallback_tuple
        )

    def _materialize_variadic_concatenation(self, unpacked: TypeList) -> UnpackType:
        """Convert a concatenation of UnpackType / items into a single UnpackType."""
        parsed_items: list[ProperType] = []
        for proper_item in map(get_proper_type, unpacked.items):
            if isinstance(proper_item, UnpackType):
                # recurse when seeing UnpackType
                proper_item = self.parse_variadic_type(proper_item)
            parsed_items.append(proper_item)

        if not parsed_items:
            # empty concatenation, return UnpackType[Never] to indicate no variadic part
            return UnpackType(UninhabitedType())

        if len(parsed_items) == 1 and isinstance(unpack := parsed_items[0], UnpackType):
            # single unpack, just return it directly
            return unpack

        # more than one unpack: cast every member as Iterable[T] and unify the T's
        item_types: list[Type] = []
        for item in parsed_items:
            if isinstance(item, UnpackType):
                iterable_type = self.context.as_iterable_type(item.type)
                item_types.append(iterable_type.args[0])
            else:
                item_types.append(item)
        unified_item_type = make_simplified_union(item_types)
        return UnpackType(self.context.make_tuple_instance_type(unified_item_type))

    def _materialize_variadic_union(self, unpacked: UnionType) -> UnpackType:
        """Convert a Union of UnpackType into a single UnpackType."""
        # Currently, Union of star args are not part of the typing spec.
        # Therefore, we need to reunify such unpackings.
        # We create an upper bound by converting each union item to an iterable,
        # and then returning the tuple unpacking *tuple[U₁ | U₂ | ... | Uₙ, ...]
        # See Also: https://discuss.python.org/t/should-unions-of-tuples-tvts-be-allowed-inside-unpack/102608

        # NOTE: We want to use set here, but we actually need stable ordering for unit tests.
        parsed_items: list[UnpackType] = []
        seen_items: set[UnpackType] = set()
        for proper_item in unpacked.proper_items:
            # unions members should all be UnpackType themselves
            assert isinstance(proper_item, UnpackType)
            parsed_item = self.parse_variadic_type(proper_item)
            if parsed_item not in seen_items:
                parsed_items.append(parsed_item)
            seen_items.add(parsed_item)

        if not parsed_items:
            return UnpackType(UninhabitedType())

        if len(parsed_items) == 1:
            return parsed_items[0]

        # more than one unpack: cast every member as Iterable[T] and unify the T's
        item_types: list[Type] = []
        for item in parsed_items:
            if isinstance(item, UnpackType):
                iterable_type = self.context.as_iterable_type(item.type)
                item_types.append(iterable_type.args[0])
            else:
                item_types.append(item)
        unified_item_type = make_simplified_union(item_types)
        return UnpackType(self.context.make_tuple_instance_type(unified_item_type))

    def _unify_multiple_unpacks(self, items: list[ProperType]) -> list[ProperType]:
        r"""If multiple UnpackType are present, unify them into a single Unpack[tuple[T, ...]]."""

        # algorithm very similar to TupleNormalForm.from_items, but now we construct a concrete
        # type.
        seen_unpacks = 0
        prefix_items: list[ProperType] = []
        unpack_items: list[ProperType] = []
        suffix_items: list[ProperType] = []

        for item in flatten_nested_tuples(items):
            if isinstance(item, UnpackType):
                seen_unpacks += 1
                unpack_items.extend(suffix_items)
                unpack_items.append(item)
                suffix_items.clear()
            elif seen_unpacks:
                suffix_items.append(item)
            else:
                prefix_items.append(item)

        if seen_unpacks <= 1:
            # we can just use the original list
            return items

        # unify all members of unpack_items into a single tuple[T, ...]
        item_types = []
        for item in unpack_items:
            if isinstance(item, UnpackType):
                iterable_type = self.context.as_iterable_type(item.type)
                item_types.append(iterable_type.args[0])
            else:
                item_types.append(item)

        unified_item_type = make_simplified_union(item_types)
        unified_unpacked = UnpackType(self.context.make_tuple_instance_type(unified_item_type))
        return [*prefix_items, unified_unpacked, *suffix_items]

    def parse_variadic_type(self, typ: AbstractUnpackType, /) -> UnpackType:
        r"""Parse the (dirty) UnpackType of a TupleNormalForm.

        A TupleNormalForm's unpack may contain the following unexpected types:

        1. UninhabitedType: indicates no variadic part
        2. TypeList: indicates concatenation of multiple variadic parts
        3. UnionType: indicates union of multiple variadic parts

        After processing with this function, the result is guaranteed to be one of:

        1. UninhabitedType: indicates no variadic part
        2. regular UnpackType content.
        """

        unpacked = get_proper_type(typ.type)

        if isinstance(unpacked, UninhabitedType):
            # this is used to indicate no variadic part
            return typ

        if isinstance(unpacked, TypeList):
            return self._materialize_variadic_concatenation(unpacked)

        elif isinstance(unpacked, UnionType):
            return self._materialize_variadic_union(unpacked)

        elif isinstance(
            unpacked, (ParamSpecType, TypeVarTupleType)
        ) or self.context.is_tuple_instance_type(unpacked):
            # already a proper element. Just return it.
            return typ

        # otherwise, cast to Iterable[T] using the solver, and then return tuple[T, ...]
        r = self.context.as_iterable_type(unpacked)
        if isinstance(r, AnyType):
            return UnpackType(self.context.make_tuple_instance_type(r))
        return UnpackType(self.context.make_tuple_instance_type(r.args[0]))


def all_tuples(types: Sequence[ProperType]) -> TypeGuard[Sequence[TupleType]]:
    """Check if all types are tuples."""
    return all(isinstance(typ, TupleType) for typ in types)


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


def normalize_finite_tuple_type(typ: FiniteTuple) -> FiniteTuple:
    """Normalize a tuple type to a FiniteTupleType."""
    t = TupleType(flatten_nested_tuples(typ.items), fallback=typ.partial_fallback)
    return FiniteTuple(t)


def is_tuple_instance_type(typ: Type) -> TypeIs[TupleInstanceType]:
    """Check if the type is an instance of `tuple[T, ...]`."""
    p_t = get_proper_type(typ)
    return isinstance(p_t, Instance) and p_t.type.fullname == "builtins.tuple"


def is_tuple_like_type(typ: Type) -> TypeIs[TupleLikeType]:
    """Check if the type is a tuple-like type."""
    p_t = get_proper_type(typ)
    return isinstance(p_t, TupleType | TypeVarTupleType) or is_tuple_instance_type(p_t)


def is_finite_tuple_type(typ: Type) -> TypeIs[FiniteTuple]:
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
