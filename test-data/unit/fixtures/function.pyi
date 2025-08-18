class object:
    def __init__(self): pass

class type: pass
class function: pass
class int: pass
class str: pass
class dict: pass

from typing import Iterable, Generic, TypeVar
_T_co = TypeVar('_T_co', covariant=True)
class tuple(Generic[_T_co], Iterable[_T_co]): pass
