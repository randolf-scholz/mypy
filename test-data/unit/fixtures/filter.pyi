# Minimal set of builtins required to work with Enums

from typing import Generic, Iterable, Self, Callable, TypeVar, Any, overload
from typing_extensions import TypeIs, TypeGuard

_T = TypeVar('_T')
_S = TypeVar('_S')

class filter(Generic[_T]):
    @overload
    def __new__(cls, function: None, iterable: Iterable[_T | None], /) -> Self: ...
    @overload
    def __new__(cls, function: Callable[[_S], TypeGuard[_T]], iterable: Iterable[_S], /) -> Self: ...
    @overload
    def __new__(cls, function: Callable[[_S], TypeIs[_T]], iterable: Iterable[_S], /) -> Self: ...
    @overload
    def __new__(cls, function: Callable[[_T], Any], iterable: Iterable[_T], /) -> Self: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> _T: ...

class function: pass
class type: pass
class int: pass
class tuple: pass
class bool(int): pass
class float: pass
class str: pass
class bool: pass
class ellipsis: pass
class dict: pass
class bytes: pass
