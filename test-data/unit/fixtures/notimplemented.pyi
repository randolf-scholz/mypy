# builtins stub used in NotImplemented related cases.
from typing import Any

class object:
    def __init__(self) -> None: pass

class type: pass
class function: pass
class bool: pass
class int: pass
class str: pass
class dict: pass

class _NotImplementedType(Any):
    __call__: NotImplemented  # type: ignore
NotImplemented: _NotImplementedType

class BaseException: pass

# region ArgumentInferContext
from typing import Mapping, Generic, Iterator, TypeVar
_Tuple_co = TypeVar('_Tuple_co', covariant=True)
class tuple(Generic[_Tuple_co]):
    def __iter__(self) -> Iterator[_Tuple_co]: pass
# endregion ArgumentInferContext
