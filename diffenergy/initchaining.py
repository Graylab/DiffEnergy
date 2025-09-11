import inspect
from typing import Any, Generic, Tuple, overload
from typing_extensions import Callable, Concatenate, ParamSpec, Protocol, TypeVar, TypeVarTuple

import functools

def compute_mro(*bases:type)->list[type]:
  return functools._c3_merge(base.mro() for base in bases)

A = TypeVarTuple("A")
B = TypeVarTuple("B")
SelfT = TypeVar("SelfT", contravariant=True)


# class Init(Protocol[SelfT, P]):
#     def __call__(__self, self: SelfT, *args: P.args, **kwds: P.kwargs) -> None:
#         ...

# class Initiable(Protocol[P]): #class with init method with paramspec P
#     def __init__(self,*args:P.args,**kwds:P.kwargs) -> None: ...

class ArgsInitiable(Protocol[*A]): #class whose init pops some args, but with no typing of the kwargs
    def __init__(self,*args: *A,**kwargs:Any) -> None: ...

class ArgsCallable(Protocol[*A]):
    def __call__(self,*args: *A, **kwargs:Any) -> None: ...

# def overinit(init: Callable[Concatenate[SelfT, P], None]) -> Init[SelfT, P]:
#     def __init__(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> None:
# # put your code here
#         init(self, *args, **kwargs)

#     return __init__

@overload
def chaininit(bases:ArgsInitiable[*B]): ...

def chaininit(*bases:ArgsInitiable):
    def decorate(init: ArgsCallable[SelfT,*A])->ArgsCallable[SelfT,*A,*B]:
        sig = inspect.signature(init)
        numarg = 0
        haskwargs = False
        for name,param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TypeError("__init__ must have finite positional arguments!")
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                numarg += 1
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                haskwargs = True
        if not haskwargs:
            raise TypeError("__init__ must have **kwargs parameter!")

        def __init__(self:SelfT,*args,**kwargs):
            init(self,*args[:numarg],**kwargs)
            
        
# from abc import ABCMeta

# class InitChain(type):
#     def __new__(cls, name, bases, attrs):
#         # customize the creation of new classes here...
#         obj = super().__new__(cls, name, bases, attrs)

#     def __init__(self, name, bases, attrs):
#         oldinit = attrs.get('__init__',None)
#         from IPython import embed; embed()
#         # perform any additional initialization here...
#         super().__init__(name, bases, attrs)



# class test(metaclass=InitChain):
    # def __init__(self,*args,**kwargs): pass


class Toplevel:
    def __init__(self,arg1,kwarg1=None) -> None:
        pass

class Mixin:
    def __init__(self,consumearg,*args,kwarg2=None,**kwargs):
        super().__init__(*args,**kwargs)

class Leaf(Toplevel):
    def __init__(self,arg2,arg3,*args,**kwargs):
        super().__init__(*args,**kwargs)

class MixedLeaf(Mixin,Toplevel):
    @overinit
    def __init__(self,arg2,arg3,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)

Leaf()
        
