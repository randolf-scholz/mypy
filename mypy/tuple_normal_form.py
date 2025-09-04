from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple, NewType, TypeGuard, TypeIs
from typing_extensions import NewType, TypeAlias as _TypeAlias, TypeGuard

from mypy.typeops import make_simplified_union
from mypy.types import (
    AnyType,
    Instance,
    ParamSpecType,
    ProperType,
    TupleType,
    Type,
    TypeList,
    TypeVarTupleType,
    UninhabitedType,
    UnionType,
    UnpackType,
    flatten_nested_tuples,
    get_proper_type,
)

if TYPE_CHECKING:
    from mypy.infer import ArgumentInferContext, TupleInstanceType


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

        # # determine the head part
        # for item in generator:
        #     if _is_empty_unpack(item):
        #         # skip empty unpacks
        #         continue
        #     elif _is_non_empty_unpack(item):
        #         body_items.append(item)
        #         break
        #     else:
        #         head_items.append(item)
        #
        # # determine the body and tail parts
        # for item in generator:
        #     if _is_empty_unpack(item):
        #         # skip empty unpacks
        #         continue
        #     elif _is_non_empty_unpack(item):
        #         body_items.extend(tail_items)
        #         body_items.append(item)
        #         tail_items.clear()
        #     else:
        #         tail_items.append(item)

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
        target_body_items: list[ProperType] = []
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
        show(f"Materializing TupleNormalForm\n\t{self}\n\t{result}")
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

        unpacked = get_proper_type(parsed_variadic_part.type)

        if isinstance(unpacked, TypeVarTupleType | ParamSpecType):
            is_empty = False
        elif isinstance(unpacked, TupleType):
            is_empty = not unpacked.proper_items
        elif self.context.is_tuple_instance_type(unpacked):
            is_empty = isinstance(unpacked.args[0], UninhabitedType)
        else:
            raise TypeError(f"unexpected type {unpacked!r}")

        if is_empty:
            assert not tnf.suffix, f"Failed to correctly parse TupleNormalForm: {tnf}"
            return TupleType([*tnf.prefix], self.context.tuple_type)

        return TupleType([*tnf.prefix, parsed_variadic_part, *tnf.suffix], self.context.tuple_type)

    def _materialize_variadic_concatenation(self, items: list[ProperType]) -> UnpackType:
        if not items:
            # return Unpack[tuple[()]]
            return UnpackType(self.context.make_tuple_type([]))
        if len(items) == 1 and isinstance(unpack := items[0], UnpackType):
            # single unpack, just return it directly
            return unpack

        # otherwise, convert every Unpack into an iterable type
        item_types: list[Type] = []
        for item in items:
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
        """Converts a dirty UnpackType into a proper UnpackType,

        by converting unexpected members (TypeList/UnionType) into proper types.
        """

        unpacked = get_proper_type(typ.type)
        if isinstance(unpacked, TypeList):
            # Convert a TypeList into a real tuple.
            parsed_items = []
            for proper_item in map(get_proper_type, unpacked.items):
                if isinstance(proper_item, UnpackType):
                    # recurse when seeing UnpackType
                    proper_item = self.parse_variadic_type(proper_item)
                parsed_items.append(proper_item)

            return self._materialize_variadic_concatenation(parsed_items)

            # if multiple unpacks are present, we unify everything into a single tuple[T, ...]
            parsed_items = self._unify_multiple_unpacks(parsed_items)

            # simplify if possible
            if len(parsed_items) == 1 and isinstance(parsed_items[0], UnpackType):
                return parsed_items[0]

            # otherwise, combine to

            tuple_result = TupleType(flatten_nested_tuples(parsed_items), self.context.tuple_type)
            return UnpackType(tuple_result)

        elif isinstance(unpacked, UnionType):
            # Currently, Union of star args are not part of the typing spec.
            # Therefore, we need to reunify such unpackings.
            # We create an upper bound by converting each union item to an iterable,
            # and then returning the tuple unpacking *tuple[U₁ | U₂ | ... | Uₙ, ...]
            # See Also: https://discuss.python.org/t/should-unions-of-tuples-tvts-be-allowed-inside-unpack/102608
            parsed_items = []
            for proper_item in unpacked.proper_items:
                if isinstance(proper_item, UnpackType):
                    # recurse when seeing UnpackType
                    proper_item = self.parse_variadic_type(proper_item)
                r = self.context.as_iterable_type(proper_item)
                parsed_items.append(r)
            assert all(self.context.is_iterable_instance_type(r) for r in parsed_items)
            union_arg = make_simplified_union([r.args[0] for r in parsed_items])
            return UnpackType(self.context.make_tuple_instance_type(union_arg))
        elif isinstance(
            unpacked, ParamSpecType | TypeVarTupleType
        ) or self.context.is_tuple_instance_type(unpacked):
            # already a proper element. Just return it.
            return typ
        # otherwise, cast to Iterable[T] using the solver, and then return tuple[T, ...]
        r = self.context.as_iterable_type(unpacked)
        if isinstance(r, AnyType):
            return UnpackType(self.context.make_tuple_instance_type(r))
        return UnpackType(self.context.make_tuple_instance_type(r.args[0]))


# def _split_tuple_type(typ: TupleType) -> TupleNormalForm:
#     r"""Split a tuple into 3 parts: head part, body part and tail part.
#
#     1. A head part which is the longest finite prefix of the tuple
#     2. A body part which covers all items from the first variable item to the last variable item
#     3. A tail part which is the longest finite suffix of the remaining tuple
#
#     If the body part is empty, the tail part is empty as well.
#     The body part, if non-empty, always starts and ends with a variable item (UnpackType).
#     Note that according to the current specification, the body part may contain at maximum
#     a single variable item (UnpackType), so the body part actually should at maximum be
#     of length 1. This implementation should still work if that specification changes in the future.
#
#     Examples:
#         - tuple[int, str] -> (tuple[int, str], tuple[()], tuple[()])
#         - tuple[int, *tuple[int, ...]] -> (tuple[int], tuple[*tuple[int, ...]], tuple[()])
#         - tuple[*tuple[int, ...], int] -> (tuple[())], tuple[*tuple[int, ...]], tuple[int])
#         - tuple[int, *tuple[int, ...], int] -> (tuple[int], tuple[*tuple[int, ...]], tuple[int])
#         - tuple[int, *tuple[int, ...], str, *tuple[str, ...], int]
#           -> (head=tuple[int], variable=tuple[*tuple[int, ...], str, *tuple[str, ...]], tail=tuple[int])
#     """
#     head_items: list[ProperType] = []
#     tail_items: list[ProperType] = []
#     body_items: list[ProperType] = []
#
#     flattened_items = flatten_nested_tuples(typ.items)
#     generator = iter(flattened_items)
#
#     # determine the head part
#     for item in generator:
#         p_t = get_proper_type(item)
#         if isinstance(p_t, UnpackType):
#             body_items.append(p_t)
#             break
#         head_items.append(p_t)
#
#     # determine the body and tail parts
#     for item in generator:
#         p_t = get_proper_type(item)
#         if isinstance(p_t, UnpackType):
#             body_items.extend(tail_items)
#             body_items.append(p_t)
#             tail_items.clear()
#         else:
#             tail_items.append(p_t)
#
#     return TupleNormalForm(head_items, body_items, tail_items)


# def split_union_of_tuple_types(types: Sequence[TupleType]) -> TupleNormalForm:
#     """Combine multiple tuple types, providing an upper bound for the union of the input tuples.
#
#     Because this function is needed in places where we may not have access to the `TypeInfo`
#     of `builtins.tuple`, construction of the resulting tuple is deferred to the caller.
#
#     Note:
#         - For the body part, callers should disregard the order of items.
#         = If all input tuples are of the same (real) size, so is the result.
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
#     heads: list[list[ProperType]]
#     bodies: list[list[ProperType]]
#     tails: list[list[ProperType]]
#     remaining_head_items: list[list[ProperType]]
#     remaining_tail_items: list[list[ProperType]]
#     remaining_body_items: list[list[ProperType]]
#     target_head_items: list[ProperType] = []
#     target_body_items: list[ProperType] = []
#     target_tail_items: list[ProperType] = []
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
#     assert len(remaining_head_items) == len(remaining_tail_items) == len(bodies)
#     remaining_body_items = [
#         [*remaining_head, *body, *remaining_tail]
#         for remaining_head, body, remaining_tail
#         in zip(remaining_head_items, bodies, remaining_tail_items)
#     ]
#
#     # 4. collected all items that will be put into the variable part
#     union_item = make_simplified_union([TypeList(body_items) for body_items in remaining_body_items])
#     target_body_items = [] if isinstance(union_item, UninhabitedType) else [union_item]
#
#     # 5. combine all parts into a TupleNormalForm
#     return TupleNormalForm(target_head_items, target_body_items, target_tail_items)


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
