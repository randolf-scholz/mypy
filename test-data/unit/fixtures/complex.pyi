# Builtins stub used for some float/complex test cases.
# Please don't add tuple to this file, it is used to test incomplete fixtures.

class object:
    def __init__(self): pass

class type: pass
class function: pass
class int: pass
class float: pass
class complex: pass
class str: pass
class dict: pass

from typing import Iterable, Generic, TypeVar
_T_co = TypeVar('_T_co', covariant=True)
class tuple(Generic[_T_co], Iterable[_T_co]): pass
