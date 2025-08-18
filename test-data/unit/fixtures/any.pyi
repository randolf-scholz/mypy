from typing import TypeVar, Iterable

T = TypeVar('T')

class int: pass
class str: pass

def any(i: Iterable[T]) -> bool: pass

class dict: pass

from typing import Iterable, Generic, TypeVar
_T_co = TypeVar('_T_co', covariant=True)
class tuple(Generic[_T_co], Iterable[_T_co]): pass
