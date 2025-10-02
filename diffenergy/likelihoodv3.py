from abc import ABC, abstractmethod
import functools
import itertools
from decorator import decorator
from typing import Callable, ClassVar, Generic, Iterable, Iterator, Literal, Mapping, Optional, Protocol, Sequence, Sized, TypeVar, TypeVarTuple, TypedDict, Union, overload
from numpy.typing import ArrayLike

import torch
from torch import NumberType, Tensor
from torchdiffeq import odeint

import numpy as np

Array = Union[np.ndarray,Tensor]

X = TypeVar("X") #X
C = TypeVar("C") #Condition

# TODO: does to_arr / from_arr ever need conditioning? E.g. for like irrep rotational representations

##### Abstract Classes / Helpers #####

#### Integrands (diffusion and ode) ####

class LikelihoodIntegrand(ABC,Generic[X,C]):

    def name(self)->str:
        return self.__class__.__name__
    
    @abstractmethod
    def diffintegrand(self,x:X,t:float,deltax:X,deltat:float,conditioning:C)->float | Array: 
        """integrand as a function of x(i), t(i), deltax(i) = x(i')-x(i), and deltat(i) = t(i')-t(i).
        Also takes additional customizable conditioning metadata."""
        ...


class ODELikelihoodIntegrand(LikelihoodIntegrand[X,C]):
    def __init__(self,to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]) -> None:
        self.to_arr = to_arr
        self.from_arr = from_arr

    @abstractmethod
    def odeintegrand(self,x:X,t:float,dx:X,dt:float,conditioning:C)->float | Array: 
        """integrand as a function of x(i), t(i), dx/di, and dt/di.
        Also takes additional customizable conditioning metadata."""
        ...

    @abstractmethod
    def shape(self,x:X)->tuple[int,...]: 
        ...

    def zero(self,x:X)->Tensor:
        xt = self.to_tensor(x)
        return torch.zeros(self.shape(x),dtype=xt.dtype,device=xt.device);

    def tensor(self,a:ArrayLike,**kwargs):
        return torch.as_tensor(a,**kwargs)

    def to_tensor(self,x:X,**kwargs):
        return self.tensor(self.to_arr(x),**kwargs)

    #convenience function - given x:X and arbitrary sequence of tensorables n (read: time followed by integrand values), turn each into a tensor
    def xntensor(self,x:X,*n:ArrayLike):
        xt = self.to_tensor(x)
        return xt,*map(functools.partial(self.tensor,device=xt.device),n)

    def diffintegrand(self, x: X, t: float, deltax: X, deltat: float, conditioning: C) -> float | Array:
        return self.odeintegrand(x, t, deltax, deltat, conditioning)



#### Integration Paths and Numerical Integration Methods ####

class IntegrablePath(ABC,Sized,Iterable[tuple[X,float]],Generic[X,C]):
    def __init__(self,to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X], conditioning: C):
        self.to_arr = to_arr
        self.from_arr = from_arr
        self.condition = conditioning

    def delta(self)->Iterable[tuple[X,float]]:
        return map(lambda xt: (self.from_arr(self.to_arr(xt[1][0]) - self.to_arr(xt[0][0])),xt[1][1]-xt[0][1]), itertools.pairwise(self))
    
    def diffintegrate(self, *integrands: LikelihoodIntegrand[X,C])->tuple[Sequence[X],Sequence[float],list[float|Array]]:
        acc:list[None|float|Array] = [None]*len(integrands)
        it = iter(self)
        (x,t) = next(it)
        accx = [x]
        acct = [t]
        for (x2,t2) in it:
            xarr,x2arr = self.to_arr(x), self.to_arr(x2)
            dxarr = x2arr-xarr
            dx = self.from_arr(dxarr)
            dt = t2-t
            
            for i,integrand in enumerate(integrands):
                I = integrand.diffintegrand(x,t,dx,dt,self.condition)
                if acc[i] is None:
                    acc[i] = I
                else:
                    acc[i] += I

            (x,t) = (x2,t2)
            accx.append(x)
            acct.append(t)

        for i in range(len(acc)): #don't think this would ever happen but /shrug
            if acc[i] is None:
                acc[i] = 0

        return accx,acct,acc

class IntegrableSequence(Sequence[tuple[X,float]],IntegrablePath[X,C]): #where path is an explicit sequence of x and t
    def __init__(self,path:Sequence[tuple[X,float]],to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C):
        self.path = path
        super().__init__(to_arr,from_arr,conditioning)
    
    @overload
    def __getitem__(self, index: int) -> tuple[X,float]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[X,float]]: ...
    def __getitem__(self, index: int|slice):
        return self.path[index]
    
    def __len__(self) -> int:
        return len(self.path)
    
class InterpolatedIntegrableSequence(IntegrableSequence[X,C]):
    def __init__(self, path:Sequence[tuple[X,float]], n_interp: int, to_arr: Callable[[X],Array], from_arr: Callable[[ArrayLike],X], conditioning:C, tmin: float = 0, tmax: float = 1):
        """n_interp of i means i-1 extra points in between each original points. n_interp of 1 is the original sequence"""
        self.n_interp = n_interp
        self.orig_sequence = path = list(path)
        it = ((torch.as_tensor(to_arr(x)),t) for x,t in path)
        x1,t1 = next(it)
        accpath = [(from_arr(x1),t1)]
        for x2,t2 in it:
            interps = torch.linspace(0,1,n_interp+1,device=x1.device,dtype=x1.dtype)[1:]
            interp_times = (1-interps)*t1 + interps*t2

            #stupid-ass manual linspace because worthless torch.linspace doesn't support vectors aaaaaa
            interps = interps.view((-1,)+(1,)*x1.ndim) #add dimensions to properly multiply arbitrarily-shaped x
            interp_points = (1-interps)*x1[None,...] + interps*x2[None,...] 
            accpath.extend(zip(map(from_arr,interp_points),interp_times))
            x1,t1 = x2,t2
        super().__init__(accpath,to_arr,from_arr,conditioning)

    
class ODEIntegrablePath(IntegrablePath[X,C],ABC):
    def __init__(self,timeschedule:Sequence[float],initial:tuple[X,float],rtol:float,atol:float,method:str,to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C):
        self.timeschedule = timeschedule
        self.initial = initial
        self.rtol = rtol
        self.atol = atol
        self.method = method
        super().__init__(to_arr,from_arr,conditioning)
            
    @abstractmethod
    def dx(self,i:float,x:X,t:float)->X:
        ...

    @abstractmethod
    def dt(self,i:float,x:X,t:float)->float:
        ...



    def tensor(self,a:ArrayLike,**kwargs):
        return torch.as_tensor(a,**kwargs)

    def to_tensor(self,x:X,**kwargs):
        return self.tensor(self.to_arr(x),**kwargs)

    #convenience function - given x:X and arbitrary sequence of tensorables n (read: time followed by integrand values), turn each into a tensor
    def xntensor(self,x:X,*n:ArrayLike):
        xt = self.to_tensor(x)
        return xt,*map(functools.partial(self.tensor,device=xt.device),n)

    #integrate dx and dt to make a path through space based on the timeschedule
    def __iter__(self) -> Iterator[tuple[X, float]]:
        def dxdtconcat(i:float,v:tuple[Tensor,Tensor]):
            x,t = v
            return self.xntensor(self.dx(i,self.from_arr(x),t.item()),self.dt(i,self.from_arr(x),t.item()))
        
        res:tuple[Tensor,Tensor] = odeint(dxdtconcat,self.xntensor(*self.initial),self.tensor(self.timeschedule), rtol=self.rtol, atol=self.atol, method=self.method)
        xs,ts = res
        ts = ts.tolist()

        return zip(map(self.from_arr,xs),ts)
    
    def __len__(self) -> int:
        return len(self.timeschedule)
    
    
    def odeintegrate(self,*integrands:ODELikelihoodIntegrand[X,C])->tuple[Sequence[X],Sequence[float],list[float|Array]]:
        def ode_func(i:float,v:tuple[Tensor,...]):
            x,t,*_ = v
            x,t = self.from_arr(x), t.item()
        
            dx,dt = self.dx(i,x,t),self.dt(i,x,t)
            return self.xntensor(dx,dt,*[integrand.odeintegrand(x,t,dx,dt,self.condition) for integrand in integrands])
        
        ## add a dimension to each integrand's zero to allow "scalar" integrands; odeint throws a fit with scalar tensors since they can't be concatenated
        res = odeint(ode_func,self.xntensor(*self.initial,*(integrand.zero(self.initial[0])[None,...] for integrand in integrands)),self.tensor(self.timeschedule),rtol=self.rtol,atol=self.atol,method=self.method)
        xs,ts,*I = res

        ## make sure to remove the dimension from each integrand
        return list(map(self.from_arr,xs)),list(map(Tensor.item,ts)),[i[-1][0] for i in I]
    

##### Concrete Classes #####

#### Integrands ####

class ScoreDivDiffIntegrand(ODELikelihoodIntegrand[X,C]):
    """Any integrand which is a function of score, divergence, and the diffusion coefficient (as a scalar function of time).
    Base Class for TotalIntegrand, TimeIntegrand, and SpaceIntegrand.
    Parameters:
    scorefn: (x,t) -> grad_xlogp(x,t))
    divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
    diffcoefffn: (t) -> g(t)
    """
    def __init__(self,scorefn:Callable[[X,float,C],Array], divfn:Callable[[X,float,C],float|Array], diffcoefffn:Callable[[float],float],to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]):
        self.scorefn = scorefn
        self.divfn = divfn
        self.diffcoefffn =diffcoefffn
        super().__init__(to_arr,from_arr)

    def shape(self, x: X) -> tuple[int, ...]:
        arr = self.to_tensor(x)
        if arr.ndim == 0:
            raise ValueError
        elif arr.ndim == 1: #single unbatched vector, scalar output
            return tuple() #scalar has shape 0. Technically this is the same as 1d_shape[:-1] but I want to make it explicit
        else:
            return arr.shape[:-1] #return all batch dimensions


class TotalIntegrand(ScoreDivDiffIntegrand[X,C]):
    """Assumes: f(x,t) = 0, G(x,t)=g(t)I_n.
    Parameters:
        scorefn: (x,t) -> grad_xlogp(x,t))
        divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
        diffcoefffn: (t) -> g(t)
        """
    def odeintegrand(self, x: X, t: float, dx: X, dt: float, conditioning:C) -> float:
        """[dlogp(x(i),t(i))/di] = grad_x(logp) dot dx/di + dlogp/dt*dt/di. Function of x, t, dx/dy, and dt/di.
        """

        gradxlogp = self.tensor(self.scorefn(x,t,conditioning)) #can be batched! Either D or BxD array
        divergence = self.divfn(x,t,conditioning) #can be batched! either scalar or size-B array
        g = self.diffcoefffn(t) #always a scalar
        g2 = g**2
        
        ##dx term: grad_x(logp) dot dx/di
        dxterm = torch.linalg.vecdot(gradxlogp,self.to_tensor(dx))

        ##divergence term: 1/2g(t)^2*divergence
        divterm = 1/2*g2*divergence*dt

        ##gradient norm term: 1/2g(t)^2*||grad_x(logp)||^2
        gradnormterm = 1/2*g2*torch.linalg.vecdot(gradxlogp,gradxlogp)*dt

        return dxterm + divterm + gradnormterm #not a float, teehee. scalar tensor though!

class TimeIntegrand(ScoreDivDiffIntegrand[X,C]):
    """Assumes: f(x,t) = 0, G(x,t)=g(t)I_n. Also assumes dx/dt = f~(t), where f~ is the appropriate flow-equivalent ODE of the SDE being modeled.
    Parameters:
        scorefn: (x,t) -> grad_xlogp(x,t))
        divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
        diffcoefffn: (t) -> g(t)
        """        
    def odeintegrand(self, x: X, t: float, dx: X, dt: float, conditioning:C) -> float:
        """[dlogp(x(i),t(i))/di] = -grad_x(f(x(t),t))dt/di [=0 since f is 0] + 1/2 g(t)^2laplacian(logp(x(t),t))dt/di. Function of x, t, dx/dy, and dt/di.
        """

        # gradxlogp = self.tensor(self.scorefn(x,t,conditioning)) #can be batched! Either D or BxD array
        divergence = self.divfn(x,t,conditioning) #can be batched! either scalar or size-B array
        g = self.diffcoefffn(t) #always a scalar
        g2 = g**2
        
        ##dx term: grad_x(logp) dot dx/di
        # dxterm = torch.linalg.vecdot(gradxlogp,self.to_tensor(dx))

        ##divergence term: 1/2g(t)^2*divergence
        divterm = 1/2*g2*divergence*dt

        ##gradient norm term: 1/2g(t)^2*||grad_x(logp)||^2
        # gradnormterm = 1/2*g2*torch.linalg.vecdot(gradxlogp,gradxlogp)*dt

        return divterm #not a float, teehee. scalar tensor though!

class SpaceIntegrand(ScoreDivDiffIntegrand[X,C]):
    """Assumes: f(x,t) = 0, G(x,t)=g(t)I_n. Also implicitly assumes that dp/dt = 0
    Parameters:
        scorefn: (x,t) -> grad_xlogp(x,t))
        divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
        diffcoefffn: (t) -> g(t)
        """
        
    def odeintegrand(self, x: X, t: float, dx: X, dt: float, conditioning: C) -> float:
        """[dlogp(x(i),t(i))/di] = grad_x(logp) dot dx/di + dlogp/dt*dt/di [=0 by assumption]. Function of x, t, dx/dy, and dt/di.
        """

        gradxlogp = self.tensor(self.scorefn(x,t,conditioning)) #can be batched! Either D or BxD array
        # divergence = self.divfn(x,t,conditioning) #can be batched! either scalar or size-B array
        # g = self.diffcoefffn(t) #always a scalar
        # g2 = g**2
        
        ##dx term: grad_x(logp) dot dx/di
        dxterm = torch.linalg.vecdot(gradxlogp,self.to_tensor(dx))

        ##divergence term: 1/2g(t)^2*divergence
        # divterm = 1/2*g2*divergence*dt

        ##gradient norm term: 1/2g(t)^2*||grad_x(logp)||^2
        # gradnormterm = 1/2*g2*torch.linalg.vecdot(gradxlogp,gradxlogp)*dt

        return dxterm #not a float, teehee. scalar tensor though!        
        

#### Paths ####

#just pass in a sequence of points. time schedule is interpreted to be tmin..tmax linearly
class UniformIntegrableSequence(IntegrableSequence[X,C]):
    def __init__(self,points:Iterable[X], to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C,tmin:float=0, tmax:float=1):
        points = list(points)
        t = np.linspace(tmin,tmax,len(points),endpoint=True)
        super().__init__(list(zip(points,t)),to_arr,from_arr,conditioning)

class InterpolatedUniformIntegrableSequence(InterpolatedIntegrableSequence[X,C]):
    def __init__(self,points:Iterable[X], n_interp: int,  to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C,tmin:float=0, tmax:float=1):
        points = list(points)
        t = np.linspace(tmin,tmax,len(points),endpoint=True)
        super().__init__(list(zip(points,t)),n_interp,to_arr,from_arr,conditioning)


class UniformODEIntegrablePath(ODEIntegrablePath[X,C]):
    def dt(self, i: float, x: X, t: float) -> float:
        return 1

class FlowEquivalentODEPath(UniformODEIntegrablePath[X,C]):
    def __init__(self, scorefn:Callable[[X,float,C],Array], diffcoefffn:Callable[[float],float], timeschedule: Sequence[float], initial: tuple[X, float], rtol: float, atol: float, method: str, to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C):
        self.scorefn = scorefn
        self.diffcoefffn = diffcoefffn
        super().__init__(timeschedule, initial, rtol, atol, method, to_arr, from_arr,conditioning)

    def dx(self, i: float, x: X, t: float) -> X:
        delta = -1/2 * self.diffcoefffn(t)**2 * self.scorefn(x,t,self.condition);

        return self.from_arr(delta);

class LinearPath(ODEIntegrablePath[X,C]):
    def __init__(self, start:tuple[X,float], end:tuple[X,float], interpolants:Sequence[float], rtol: float, atol: float, method: str, to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C):
        super().__init__(interpolants,start,rtol,atol,method,to_arr,from_arr,conditioning)
        self.end = end
        self.mini = interpolants[0]
        self.maxi = interpolants[-1]
        self.di = self.maxi-self.mini
        self.xslope = self.from_arr((self.to_arr(end[0])-self.to_arr(start[0]))/self.di)
        self.tslope = (end[1]-start[1])/self.di
        

    def dx(self, i: float, x: X, t: float) -> X:
        return self.xslope
    
    def dt(self, i: float, x: X, t: float) -> float:
        return self.tslope
    
    def __iter__(self) -> Iterator[tuple[X, float]]: #no need for integration
        x0,t0 = self.initial; x0 = self.to_arr(x0)
        x1,t1 = self.end; x1 = self.to_arr(x1)
        for i in self.timeschedule:
            interp = (i-self.mini)/self.di
            yield (self.from_arr((1-interp)*x0 + interp*x1),(1-interp)*t0+interp*t1)

class LinearizedFlowPath(LinearPath[X,C]):
    def __init__(self, scorefn:Callable[[X,float,C],Array], diffcoefffn:Callable[[float],float], timeschedule: Sequence[float], initial: tuple[X, float], rtol: float, atol: float, method: str, to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],conditioning:C,interpolants:Optional[Sequence[float]]=None):
        self.flowpath = FlowEquivalentODEPath(scorefn,diffcoefffn,timeschedule,initial,rtol,atol,method,to_arr,from_arr,conditioning)
        self.path = list(self.flowpath)
        end = self.path[-1]
        super().__init__(initial,end,interpolants or timeschedule,rtol,atol,method,to_arr,from_arr,conditioning);



def brownian_bridge(timepoints:Sequence[float], in_shape:tuple[int,...]|int,sigma=1):
    if not isinstance(in_shape,tuple):
        in_shape = (in_shape,)
    t0, t1 = timepoints[0],timepoints[-1]
    times = (np.subtract(timepoints,t0))/(t1-t0) #make times start at 0 and end at 12
    Dt = np.diff(times)
    Dt_sqrt = np.sqrt(Dt)
    B = np.empty((len(times),) + in_shape, dtype=np.float32)
    B[0] = np.zeros(in_shape)
    for i in range(len(times) - 2):
         xi = np.random.randn(*in_shape) * sigma * Dt_sqrt[i]
         B[i + 1] = B[i] * (1 - Dt[i] / (1 - times[i])) + xi
    B[-1] = 0                                                 
    return B

class PerturbedPath(IntegrableSequence[X,C]):
    def __init__(self, path: IntegrablePath[X,C], perturbation_schedule:Literal["uniform","data"], sigma: float, conditioning:C):
        points = list(path)
        shape = path.to_arr(points[0][0]).shape
        if perturbation_schedule == "data":
            times = [torch.as_tensor(point[1]).item() for point in points]
        elif perturbation_schedule == "uniform":
            times = np.linspace(0,1,len(points))
        else:
            raise ValueError(perturbation_schedule)
        offsets = brownian_bridge(times, shape, sigma=sigma)

        super().__init__([(x + offset, t) for (x,t),offset in zip(points,offsets)],path.to_arr,path.from_arr,conditioning)




#### Likelihood Calculation ####

_I = TypeVar("_I") # id type
def _run_likelihood(method:Literal['diff','ode'],id:_I,path:IntegrablePath[X,C],integrands:Sequence[LikelihoodIntegrand[X,C]]):

    if method == 'diff':
        trajectory, times, deltas = path.diffintegrate(*integrands)
    elif method == 'ode':
        if not isinstance(path,ODEIntegrablePath):
            raise ValueError(f"Path {path} is not ODEIntegrable! Please use an ODEIntegrable path or set the integral_type to 'diff' to use the path in euclidean mode");
        trajectory, times, deltas = path.odeintegrate(*integrands)

    ##Since we assume the path goes from unknown to known, we use the last data point for the prior and negate the delta
    integrand_results:dict[str,float|ArrayLike] = {integrand.name(): torch.Tensor.tolist(torch.as_tensor(-delta)) for integrand,delta in zip(integrands,deltas)}
    # prior_endpoint:tuple[X,float] = trajectory[-1]
    # prior_results:dict[str,float|ArrayLike] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*prior_endpoint))) for name,prior_fn in prior_fns}

    return (id,trajectory, times, integrand_results)

def run_diff_likelihood(id:_I,path:IntegrablePath[X,C],integrands:Sequence[LikelihoodIntegrand[X,C]]):
    return _run_likelihood('diff',id,path,integrands)

def run_ode_likelihood(id:_I,path:ODEIntegrablePath[X,C],integrands:Sequence[ODELikelihoodIntegrand[X,C]]):
    return _run_likelihood('ode',id,path,integrands)

try:
    from ray.util.joblib import register_ray
    from ray.util.joblib.ray_backend import RayBackend
    RayBackend.supports_return_generator = True
    register_ray()
except ImportError:
    pass

# Parallelism with joblib, w/ ray for keeping things in the cluster. remote_kwargs are passed to each actor and can specify resource requirements;
# set the ray cluster args to impose limits on the cluster based on the slurm spec.
_T = TypeVarTuple("_T")
_R = TypeVar("_R")
def istarmap_joblib(func:Callable[[*_T],_R],starargs:Iterable[tuple[*_T]],parallel:bool,remote_kwargs:dict):
    if parallel:
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend("ray",ray_remote_args=remote_kwargs):
            print("pre-parallel")
            res:Iterable[_R] = Parallel(return_as="generator")(delayed(func)(*args) for args in starargs)
            print("post-parallel")
            print(res,type(res))
            yield from res
    else:
        yield from (func(*args) for args in starargs)


def run_diff_likelihoods(paths:Iterable[tuple[_I,IntegrablePath[X,C]]],integrands:Sequence[LikelihoodIntegrand[X,C]],parallel=False,remote_kwargs={}):
    yield from istarmap_joblib(run_diff_likelihood,((id,path,integrands) for id,path in paths),parallel,remote_kwargs)

def run_ode_likelihoods(paths:Iterable[tuple[_I,ODEIntegrablePath[X,C]]],integrands:Sequence[ODELikelihoodIntegrand[X,C]],parallel=False,remote_kwargs={}):
    yield from istarmap_joblib(run_ode_likelihood,((id,path,integrands) for id,path in paths),parallel,remote_kwargs)