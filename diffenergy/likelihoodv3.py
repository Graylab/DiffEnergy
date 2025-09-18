from abc import ABC, abstractmethod
import functools
import itertools
from decorator import decorator
from typing import Callable, ClassVar, Generic, Iterable, Iterator, Literal, Mapping, Optional, Protocol, Sequence, Sized, TypeVar, TypedDict, Union, overload
from numpy.typing import ArrayLike

import torch
from torch import NumberType, Tensor
from torchdiffeq import odeint

import numpy as np

Array = Union[np.ndarray,Tensor]

X = TypeVar("X")

##### Abstract Classes / Helpers #####

#### Integrands (diffusion and ode) ####

class LikelihoodIntegrand(ABC,Generic[X]):

    def name(self)->str:
        return self.__class__.__name__
    
    @abstractmethod
    def diffintegrand(self,x:X,t:float,deltax:X,deltat:float)->float | Array: 
        """integrand as a function of x(i), t(i), deltax(i) = x(i')-x(i), and deltat(i) = t(i')-t(i)"""
        ...


class ODELikelihoodIntegrand(LikelihoodIntegrand[X]):
    def __init__(self,to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]) -> None:
        self.to_arr = to_arr
        self.from_arr = from_arr

    @abstractmethod
    def odeintegrand(self,x:X,t:float,dx:X,dt:float)->float | Array: 
        """integrand as a function of x(i), t(i), dx/di, and dt/di"""
        ...


    def tensor(self,a:ArrayLike,**kwargs):
        return torch.as_tensor(a,**kwargs)

    def to_tensor(self,x:X,**kwargs):
        return self.tensor(self.to_arr(x),**kwargs)

    #convenience function - given x:X and arbitrary sequence of tensorables n (read: time followed by integrand values), turn each into a tensor
    def xntensor(self,x:X,*n:ArrayLike):
        xt = self.to_tensor(x)
        return xt,*map(functools.partial(self.tensor,device=xt.device),n)

    def diffintegrand(self, x: X, t: float, deltax: X, deltat: float) -> float | Array:
        I = self.odeintegrand(x, t, deltax, deltat)
        return I*deltat

#### Integration Paths and Numerical Integration ####

class IntegrablePath(ABC,Sized,Iterable[tuple[X,float]],Generic[X]):
    def __init__(self,to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]):
        self.to_arr = to_arr
        self.from_arr = from_arr

    def delta(self)->Iterable[tuple[X,float]]:
        return map(lambda xt: (self.from_arr(self.to_arr(xt[1][0]) - self.to_arr(xt[0][0])),xt[1][1]-xt[0][1]), itertools.pairwise(self))
    
    def diffintegrate(self, *integrands: LikelihoodIntegrand[X])->tuple[Sequence[tuple[X,float]],list[float|Array]]:
        acc:list[None|float|Array] = [None]*len(integrands)
        it = iter(self)
        (x,t) = next(it)
        accpath = [(x,t)]
        for (x2,t2) in it:
            xarr,x2arr = self.to_arr(x), self.to_arr(x2)
            dxarr = x2arr-xarr
            dx = self.from_arr(dxarr)
            dt = t2-t
            
            for i,integrand in enumerate(integrands):
                I = integrand.diffintegrand(x,t,dx,dt)
                if acc[i] is None:
                    acc[i] = I
                else:
                    acc[i] += I

            (x,t) = (x2,t2)
            accpath.append((x,t))
        
        for i in range(len(acc)): #don't think this would ever happen but /shrug
            if acc[i] is None:
                acc[i] = 0

        return accpath,acc

class IntegrableSequence(Sequence[tuple[X,float]],IntegrablePath[X]): #where path is an explicit sequence of x and t
    def __init__(self,path:Sequence[tuple[X,float]],to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],):
        self.path = path
        super().__init__(to_arr,from_arr)
    
    @overload
    def __getitem__(self, index: int) -> tuple[X,float]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[X,float]]: ...
    def __getitem__(self, index: int|slice):
        return self.path[index]
    
    def __len__(self) -> int:
        return len(self.path)
    
class ODEIntegrablePath(IntegrablePath[X],ABC):
    def __init__(self,timeschedule:Sequence[float],initial:tuple[X,float],rtol:float,atol:float,method:str,to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]):
        self.timeschedule = timeschedule
        self.initial = initial
        self.rtol = rtol
        self.atol = atol
        self.method = method
        super().__init__(to_arr,from_arr)
            
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
    
    
    def odeintegrate(self,*integrands:ODELikelihoodIntegrand[X])->tuple[Sequence[tuple[X,float]],list[float|Array]]:
        def ode_func(i:float,v:tuple[Tensor,...]):
            x,t,*_ = v
            
            x,t = self.from_arr(x), t.item()
            

            dx,dt = self.dx(i,x,t),self.dt(i,x,t)
            return self.xntensor(dx,dt,*[integrand.odeintegrand(x,t,dx,dt) for integrand in integrands])
        
        res = odeint(ode_func,self.xntensor(*self.initial,*([0]*len(integrands))),self.tensor(self.timeschedule),rtol=self.rtol,atol=self.atol,method=self.method)
        xs,ts,*I = res

        accpath = zip(map(self.from_arr,xs),map(Tensor.item,ts))

        return list(accpath),[i[-1].item() for i in I]
    

##### Concrete Classes #####

#### Integrands ####

class ScoreDivDiffIntegrand(ODELikelihoodIntegrand[X]):
    """Any integrand which is a function of score, divergence, and the diffusion coefficient (as a scalar function of time).
    Base Class for TotalIntegrand, TimeIntegrand, and SpaceIntegrand.
    Parameters:
    scorefn: (x,t) -> grad_xlogp(x,t))
    divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
    diffcoefffn: (t) -> g(t)
    """
    def __init__(self,scorefn:Callable[[X,float],Array], divfn:Callable[[X,float],float], diffcoefffn:Callable[[float],float],to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]):
        self.scorefn = scorefn
        self.divfn = divfn
        self.diffcoefffn =diffcoefffn
        super().__init__(to_arr,from_arr)


class TotalIntegrand(ScoreDivDiffIntegrand[X]):
    """Assumes: f(x,t) = 0, G(x,t)=g(t)I_n.
    Parameters:
        scorefn: (x,t) -> grad_xlogp(x,t))
        divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
        diffcoefffn: (t) -> g(t)
        """
    def odeintegrand(self, x: X, t: float, dx: X, dt: float) -> float:
        """[dlogp(x(i),t(i))/di] = grad_x(logp) dot dx/di + dlogp/dt*dt/di. Function of x, t, dx/dy, and dt/di.
        """

        gradxlogp = self.scorefn(x,t)
        divergence = self.divfn(x,t)
        g = self.diffcoefffn(t)
        g2 = g**2
        
        ##dx term: grad_x(logp) dot dx/di
        dxterm = torch.dot(self.tensor(gradxlogp),self.to_tensor(dx))

        ##divergence term: 1/2g(t)^2*divergence
        divterm = 1/2*g2*divergence

        ##gradient norm term: 1/2g(t)^2*||grad_x(logp)||^2
        gradnormterm = 1/2*g2*torch.dot(self.tensor(gradxlogp),self.tensor(gradxlogp))

        return dxterm + divterm + gradnormterm #not a float, teehee. scalar tensor though!

class TimeIntegrand(ScoreDivDiffIntegrand[X]):
    """Assumes: f(x,t) = 0, G(x,t)=g(t)I_n. Also assumes dx/dt = f~(t), where f~ is the appropriate flow-equivalent ODE of the SDE being modeled.
    Parameters:
        scorefn: (x,t) -> grad_xlogp(x,t))
        divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
        diffcoefffn: (t) -> g(t)
        """        
    def odeintegrand(self, x: X, t: float, dx: X, dt: float) -> float:
        """[dlogp(x(i),t(i))/di] = -grad_x(f(x(t),t))dt/di [=0 since f is 0] + 1/2 g(t)^2laplacian(logp(x(t),t))dt/di. Function of x, t, dx/dy, and dt/di.
        """

        # gradxlogp = self.scorefn(x,t)
        divergence = self.divfn(x,t)
        g = self.diffcoefffn(t)
        g2 = g**2
        
        # ##dx term: grad_x(logp) dot dx/di
        # dxterm = torch.dot(self.tensor(gradxlogp),self.to_tensor(dx))

        ##divergence term: 1/2g(t)^2*divergence
        divterm = 1/2*g2*divergence

        # ##gradient norm term: 1/2g(t)^2*||grad_x(logp)||^2
        # gradnormterm = 1/2*g2*torch.dot(self.tensor(gradxlogp),self.tensor(gradxlogp))

        return divterm #not a float, teehee. scalar tensor though!

class SpaceIntegrand(ScoreDivDiffIntegrand[X]):
    """Assumes: f(x,t) = 0, G(x,t)=g(t)I_n. Also implicitly assumes that dp/dt = 0
    Parameters:
        scorefn: (x,t) -> grad_xlogp(x,t))
        divfn: (x,t) -> laplacian_x(logp(x,t)) = sum((d/dx_i)^2 logp(x,t))
        diffcoefffn: (t) -> g(t)
        """
        
    def odeintegrand(self, x: X, t: float, dx: X, dt: float) -> float:
        """[dlogp(x(i),t(i))/di] = grad_x(logp) dot dx/di + dlogp/dt*dt/di [=0 by assumption]. Function of x, t, dx/dy, and dt/di.
        """

        gradxlogp = self.scorefn(x,t)
        # divergence = self.divfn(x,t)
        # g = self.diffcoefffn(t)
        # g2 = g**2
        
        ##dx term: grad_x(logp) dot dx/di
        dxterm = torch.dot(self.tensor(gradxlogp),self.to_tensor(dx))

        ##divergence term: 1/2g(t)^2*divergence
        # divterm = 1/2*g2*divergence

        # ##gradient norm term: 1/2g(t)^2*||grad_x(logp)||^2
        # gradnormterm = 1/2*g2*torch.dot(self.tensor(gradxlogp),self.tensor(gradxlogp))

        return dxterm #not a float, teehee. scalar tensor though!        
        

#### Paths ####

#just pass in a sequence of points. time schedule is interpreted to be tmin..tmax linearly
class UniformIntegrableSequence(IntegrableSequence[X]):
    def __init__(self,points:Iterable[X], to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],tmin:float=0, tmax:float=1):
        points = list(points)
        t = np.linspace(tmin,tmax,len(points),endpoint=True)
        super().__init__(list(zip(points,t)),to_arr,from_arr)

class InterpolatedUniformIntegrableSequence(UniformIntegrableSequence[X]):    
    def __init__(self, points: Iterable[X], n_interp: int, to_arr: Callable[[X],Array], from_arr: Callable[[ArrayLike],X], tmin: float = 0, tmax: float = 1):
        """n_interp of i means i-1 extra points in between each original points. n_interp of 1 is the original sequence"""
        self.n_interp = n_interp
        self.orig_sequence = points = list(points)
        it = map(torch.as_tensor,map(to_arr,points))
        x1 = next(it)
        accpath = [x1]
        for x2 in it:
            interps = torch.linspace(0,1,n_interp+1,device=x1.device,dtype=x1.dtype)[1:]
            interp_points = (1-interps[:,None])*x1[None,...] + interps[:,None]*x2[None,...] #stupid-ass manual linspace because worthless torch.linspace doesn't support vectors aaaaaa
            accpath.extend(interp_points)
        interp = torch.stack(accpath,dim=0)
        assert interp.shape[0] == (len(points)-1)*(n_interp)+1, interp.shape[0]
        super().__init__(map(from_arr,interp),to_arr,from_arr)



class UniformODEIntegrablePath(ODEIntegrablePath[X]):
    def dt(self, i: float, x: X, t: float) -> float:
        return 1

class FlowEquivalentODEPath(UniformODEIntegrablePath[X]):
    def __init__(self, scorefn:Callable[[X,float],Array], diffcoefffn:Callable[[float],float], timeschedule: Sequence[float], initial: tuple[X, float], rtol: float, atol: float, method: str, to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]):
        self.scorefn = scorefn
        self.diffcoefffn = diffcoefffn
        super().__init__(timeschedule, initial, rtol, atol, method, to_arr, from_arr)

    def dx(self, i: float, x: X, t: float) -> X:
        delta = -1/2 * self.diffcoefffn(t)**2 * self.scorefn(x,t);

        return self.from_arr(delta);

class LinearPath(ODEIntegrablePath[X]):
    def __init__(self, start:tuple[X,float], end:tuple[X,float], interpolants:Sequence[float], rtol: float, atol: float, method: str, to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X]):
        super().__init__(interpolants,start,rtol,atol,method,to_arr,from_arr)
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

class LinearizedFlowPath(LinearPath[X]):
    def __init__(self, scorefn:Callable[[X,float],Array], diffcoefffn:Callable[[float],float], timeschedule: Sequence[float], initial: tuple[X, float], rtol: float, atol: float, method: str, to_arr:Callable[[X],Array],from_arr:Callable[[ArrayLike],X],interpolants:Optional[Sequence[float]]=None):
        self.flowpath = FlowEquivalentODEPath(scorefn,diffcoefffn,timeschedule,initial,rtol,atol,method,to_arr,from_arr)
        self.path = list(self.flowpath)
        end = self.path[-1]
        super().__init__(initial,end,interpolants or timeschedule,rtol,atol,method,to_arr,from_arr);
    










#### Likelihood Calculation ####

def _run_likelihood(method:Literal['diff','ode'],id:str,path:IntegrablePath[X],integrands:Sequence[LikelihoodIntegrand[X]],prior_fns:Iterable[tuple[str,Callable[[X,float],float|Array]]]):

    if method == 'diff':
        trajectory, deltas = path.diffintegrate(*integrands)
    elif method == 'ode':
        if not isinstance(path,ODEIntegrablePath):
            raise ValueError(f"Path {path} is not ODEIntegrable! Please use an ODEIntegrable path or set the integral_type to 'diff' to use the path in euclidean mode");
        trajectory, deltas = path.odeintegrate(*integrands)

    ##Since we assume the path goes from unknown to known, we use the last data point for the prior and negate the delta
    integrand_results:dict[str,float|ArrayLike] = {integrand.name(): torch.Tensor.tolist(torch.as_tensor(-delta)) for integrand,delta in zip(integrands,deltas)}
    prior_results:dict[str,float|ArrayLike] = {name:torch.Tensor.tolist(torch.as_tensor(prior_fn(*trajectory[-1]))) for name,prior_fn in prior_fns}


    return (id,prior_results,integrand_results)

def run_diff_likelihood(id:str,path:IntegrablePath[X],integrands:Sequence[LikelihoodIntegrand[X]],prior_fns:Iterable[tuple[str,Callable[[X,float],float|Array]]]):
    return _run_likelihood('diff',id,path,integrands,prior_fns)

def run_ode_likelihood(id:str,path:ODEIntegrablePath[X],integrands:Sequence[ODELikelihoodIntegrand[X]],prior_fns:Iterable[tuple[str,Callable[[X,float],float|Array]]]):
    return _run_likelihood('ode',id,path,integrands,prior_fns)


def run_diff_likelihoods(paths:Iterable[tuple[str,IntegrablePath[X]]],integrands:Sequence[LikelihoodIntegrand[X]],prior_fns:Iterable[tuple[str,Callable[[X,float],float|Array]]]):
    yield from (run_diff_likelihood(id,path,integrands,prior_fns) for id,path in paths)

def run_ode_likelihoods(paths:Iterable[tuple[str,ODEIntegrablePath[X]]],integrands:Sequence[ODELikelihoodIntegrand[X]],prior_fns:Iterable[tuple[str,Callable[[X,float],float|Array]]]):
    yield from (run_ode_likelihood(id,path,integrands,prior_fns) for id,path in paths)