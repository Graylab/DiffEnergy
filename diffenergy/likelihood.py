from abc import ABC, abstractmethod
import functools
import itertools
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, Optional, Sequence, Sized, TypeVar, Union, overload, override
from numpy.typing import ArrayLike

import torch
from torch import Tensor
from torchdiffeq import odeint

import numpy as np



X = TypeVar("X") #X
C = TypeVar("C") #Condition

# TODO: does to_arr / from_arr ever need conditioning? E.g. for like irrep rotational representations

##### Abstract Classes / Helpers #####

#### Integrands (diffusion and ode) ####
# TODO: REFACTOR DIFFINTEGRAND/ODEINTEGRAND TO RETURN A TUPLE OF F_x, T_x AND THEN DOT THEM DURING INTEGRATION.
## ^ maybe wait until we figure out riemannian diffusion to do this, though...
class LikelihoodIntegrand(ABC,Generic[X,C]):

    def name(self)->str:
        return self.__class__.__name__
    
    @abstractmethod
    def diffintegrand(self,x:X,t:float,deltax:X,deltat:float,conditioning:C)->float | Tensor: 
        """integrand as a function of x(i), t(i), deltax(i) = x(i')-x(i), and deltat(i) = t(i')-t(i).
        Also takes additional customizable conditioning metadata."""
        ...

    @abstractmethod
    def zero(self,x:X)->float|Tensor: ...


class ODELikelihoodIntegrand(LikelihoodIntegrand[X,C]):
    def __init__(self,to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X]) -> None:
        self.to_arr = to_arr
        self.from_arr = from_arr

    @abstractmethod
    def odeintegrand(self,x:X,t:float,dx:X,dt:float,conditioning:C)->float | Tensor: 
        """integrand as a function of x(i), t(i), dx/di, and dt/di.
        Also takes additional customizable conditioning metadata."""
        ...

    @abstractmethod
    def shape(self,x:X)->tuple[int,...]: 
        ...

    def zero(self,x:X)->Tensor:
        xt = self.to_tensor(x)
        return torch.zeros(self.shape(x),dtype=xt.dtype,device=xt.device)

    def tensor(self,a:ArrayLike,**kwargs):
        return torch.as_tensor(a,**kwargs)

    def to_tensor(self,x:X,**kwargs):
        return self.tensor(self.to_arr(x),**kwargs)

    #convenience function - given x:X and arbitrary sequence of tensorables n (read: time followed by integrand values), turn each into a tensor
    def xntensor(self,x:X,*n:ArrayLike):
        xt = self.to_tensor(x)
        return xt,*map(functools.partial(self.tensor,device=xt.device),n)

    def diffintegrand(self, x: X, t: float, deltax: X, deltat: float, conditioning: C) -> float | Tensor:
        return self.odeintegrand(x, t, deltax, deltat, conditioning)



#### Integration Paths and Numerical Integration Methods ####

class IntegrablePath(ABC,Sized,Iterable[tuple[X,float]],Generic[X,C]):
    def __init__(self,to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X], conditioning: C, method:str, methodargs:dict[str,Any]):
        self.to_arr = to_arr
        self.from_arr = from_arr
        self.condition = conditioning
        self.method = method
        self.methodargs = methodargs

    def delta(self)->Iterable[tuple[X,float]]:
        return map(lambda xt: (self.from_arr(self.to_arr(xt[1][0]) - self.to_arr(xt[0][0])),xt[1][1]-xt[0][1]), itertools.pairwise(self))
    
    def diffintegrate(self, *integrands: LikelihoodIntegrand[X,C], accumulate=True)->tuple[Sequence[X],Sequence[float],list[list[float|Tensor]]]:
        it = iter(self)
        (x,t) = next(it)
        likelihoods = [i.zero(x) for i in integrands]
        
        acc:list[list[float|Tensor]] = [[l] for l in likelihoods] if accumulate else []
        accx = [x] if accumulate else []
        acct = [t] if accumulate else []

        for (x2,t2) in it:
            xarr,x2arr = self.to_arr(x), self.to_arr(x2)
            dxarr = x2arr-xarr
            dx = self.from_arr(dxarr)
            dt = t2-t
            
            if self.method == 'euler':
                for i,integrand in enumerate(integrands): likelihoods[i] += integrand.diffintegrand(x,t,dx,dt,self.condition)
            elif self.method == 'euler_backward':
                #NOTE this is technically not backwards euler. it's just the same dumb euler but using the other endpoint for the riemann sum
                for i,integrand in enumerate(integrands): likelihoods[i] += integrand.diffintegrand(x2,t2,dx,dt,self.condition)
            elif self.method == 'trapezoid': # now we're getting somewhere. 
                # TODO: refactor this using smarter diffintegrand so we are guaranteed to not recalculate the score every time!
                # (once I figure out the better X system for riemannian diffusion, at least). For now, just betting on caching

                #do all of the previoius point before asking for the next to prevent cache invalidation
                I1 = [integrand.diffintegrand(x,t,dx,dt,self.condition) for integrand in integrands] 
                I2 = [integrand.diffintegrand(x2,t2,dx,dt,self.condition) for integrand in integrands]

                for i,(i1,i2) in enumerate(zip(I1,I2)): likelihoods[i] += (i1 + i2)/2
            else:
                raise ValueError(self.method)

            (x,t) = (x2,t2)
            if accumulate:
                accx.append(x)
                acct.append(t)
                [accl.append(l) for accl,l in zip(acc,likelihoods)]
        
        if accumulate:
            return accx,acct,acc
        else:
            return [x],[t],[[l] for l in likelihoods] #keep return type the same


class IntegrableSequence(Sequence[tuple[X,float]],IntegrablePath[X,C]): #where path is an explicit sequence of x and t
    def __init__(self,path:Iterable[tuple[X,float]],to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any]):
        if not isinstance(path,Sequence):
            path = list(path)
        self.path = path
        super().__init__(to_arr,from_arr,conditioning,method,methodargs)
    
    @overload
    def __getitem__(self, index: int) -> tuple[X,float]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[X,float]]: ...
    def __getitem__(self, index: int|slice):
        return self.path[index]
    
    def __len__(self) -> int:
        return len(self.path)
    
class InterpolatedIntegrableSequence(IntegrableSequence[X,C]):
    def __init__(self,  n_interp: int, path:Iterable[tuple[X,float]], to_arr: Callable[[X],Tensor], from_arr: Callable[[ArrayLike],X], conditioning:C, method:str, methodargs:dict[str,Any], tmin: float = 0, tmax: float = 1):
        """n_interp of i means i-1 extra points in between each original points. n_interp of 1 is the original sequence"""
        if not isinstance(path,Sequence):
            path = list(path)
        assert n_interp >= 1
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
        super().__init__(accpath,to_arr,from_arr,conditioning,method,methodargs)

    
class ODEIntegrablePath(IntegrablePath[X,C],ABC):
    def __init__(self,timeschedule:Sequence[float],initial:tuple[X,float],to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any]):
        self.timeschedule = timeschedule
        self.initial = initial
        methodargs = dict(**methodargs)
        
        try:
            self.rtol = methodargs.pop('rtol')
        except KeyError:
            raise ValueError("ODE Integration missing required parameter: \"rtol\"")
        try:
            self.atol = methodargs.pop('atol')
        except KeyError:
            raise ValueError("ODE Integration missing required parameter: \"atol\"")
        
        super().__init__(to_arr,from_arr,conditioning,method,methodargs)
            
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
        
        res:tuple[Tensor,Tensor] = odeint(dxdtconcat,self.xntensor(*self.initial),self.tensor(self.timeschedule), rtol=self.rtol, atol=self.atol, method=self.method, options=self.methodargs)
        xs,ts = res
        ts = ts.tolist()

        return zip(map(self.from_arr,xs),ts)
    
    def __len__(self) -> int:
        return len(self.timeschedule)
    
    
    def odeintegrate(self,*integrands:ODELikelihoodIntegrand[X,C], accumulate=True)->tuple[Sequence[X],Sequence[float],list[Sequence[float|Tensor]]]:
        def ode_func(i:float,v:tuple[Tensor,...]):
            x,t,*_ = v
            x,t = self.from_arr(x), t.item()
        
            dx,dt = self.dx(i,x,t),self.dt(i,x,t)
            return self.xntensor(dx,dt,*[integrand.odeintegrand(x,t,dx,dt,self.condition) for integrand in integrands])
        
        ## add a null dimension to each integrand's zero ([None,...]) to allow "scalar" integrands; odeint throws a fit with scalar tensors since they can't be concatenated
        res = odeint(ode_func,self.xntensor(*self.initial,*(integrand.zero(self.initial[0])[None,...] for integrand in integrands)),self.tensor(self.timeschedule), rtol=self.rtol, atol=self.atol, method=self.method, options=self.methodargs)
        xs,ts,*I = res


        if accumulate:
            ## make sure to remove the null dimension from each integrand via [...,0]
            return list(map(self.from_arr,xs)),list(map(Tensor.item,ts)),[i[:,0] for i in I]
        else:
            ## make sure to remove the null dimension from each integrand via [...,0]
            return [self.from_arr(xs[-1])],[ts[-1].item()],[i[-1,0] for i in I] #keep return type consistent
    

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
    def __init__(self,scorefn:Callable[[X,float,C],Tensor], divfn:Callable[[X,float,C],float|Tensor], diffcoefffn:Callable[[float],float],to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X]):
        self.scorefn = scorefn
        self.divfn = divfn
        self.diffcoefffn = diffcoefffn
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
    def odeintegrand(self, x: X, t: float, dx: X, dt: float, conditioning:C) -> float | Tensor:
        """[dlogp(x(i),t(i))/di] = grad_x(logp) dot dx/di + dlogp/dt*dt/di. Function of x, t, dx/di, and dt/di.
        """

        gradxlogp = self.tensor(self.scorefn(x,t,conditioning)).detach() #can be batched! Either D or BxD array
        divergence = self.tensor(self.divfn(x,t,conditioning)).detach() #can be batched! either scalar or size-B array
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
    def odeintegrand(self, x: X, t: float, dx: X, dt: float, conditioning:C) -> float | Tensor:
        """[dlogp(x(i),t(i))/di] = -grad_x(f(x(t),t))dt/di [=0 since f is 0] + 1/2 g(t)^2laplacian(logp(x(t),t))dt/di. Function of x, t, dx/di, and dt/di.
        """

        # gradxlogp = self.tensor(self.scorefn(x,t,conditioning)).detach() #can be batched! Either D or BxD array
        divergence = self.tensor(self.divfn(x,t,conditioning)).detach() #can be batched! either scalar or size-B array
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
        
    def odeintegrand(self, x: X, t: float, dx: X, dt: float, conditioning: C) -> float | Tensor:
        """[dlogp(x(i),t(i))/di] = grad_x(logp) dot dx/di + dlogp/dt*dt/di [=0 by assumption]. Function of x, t, dx/di, and dt/di.
        """

        gradxlogp = self.tensor(self.scorefn(x,t,conditioning)).detach() #can be batched! Either D or BxD array
        # divergence = self.tensor(self.divfn(x,t,conditioning)).detach() #can be batched! either scalar or size-B array
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

# Reverse SDE Path generator. Only goes from tnoise=1 to tdata=0, should be equivalent to neural network inference 
# (replaces the euler_maruyama sampler for trajectory generation)
class ReverseSDEPath(IntegrablePath[X,C]):
    def __init__(self, scorefn:Callable[[X,float,C],Tensor], diffcoefffn:Callable[[float],float], noise_scale:float, times: Sequence[float], initial: X, to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X], conditioning:C, method:str, methodargs:dict[str,Any]):
        self.scorefn = scorefn
        self.diffcoefffn = diffcoefffn
        self.times = times
        self.initial = initial
        self.noise_scale = noise_scale #should be 1 by default. allows you to forcibly turn up/down the noise during sampling. TODO: make this a proper schedule?
        
        super().__init__(to_arr,from_arr,conditioning,method,methodargs)
    
    def __len__(self) -> int:
        return len(self.times)

    def __iter__(self) -> Iterator[tuple[X, float]]:
        x = self.initial
        time_iter = iter(self.times)
        time_step = next(time_iter)
        for t2 in time_iter:
            yield (x,time_step)
            dt = (t2 - time_step) #note that this means dt is negative!!! so we've gotta negate it when we use it
            assert dt < 0, "dt must be negative for reverse SDE sampling! Make sure you use a time schedule which decreases monotonically"
            g = self.diffcoefffn(time_step)
            score = self.scorefn(x,time_step,self.condition).detach()


            with torch.no_grad():
                x_arr = self.to_arr(x)
                dx = - (g**2) * score * dt #negative because dt is negative!
                dx += self.noise_scale * torch.sqrt(-dt) * g * torch.randn_like(x_arr) #negate dt so real sqrt
                x_arr = x_arr + dx #don't do it in place! these tensors don't get cloned. (maybe they should?)
                x = self.from_arr(x_arr)


            time_step = t2

        yield (x,time_step)

#the reverse of ReverseSDE path. Follows a random diffusion trajectory from a starting point.
class ForwardSDEPath(IntegrablePath[X,C]):
    def __init__(self, diffcoefffn:Callable[[float],float], noise_scale:float, times: Sequence[float], initial: X, to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X], conditioning:C, method:str, methodargs:dict[str,Any]):
        self.diffcoefffn = diffcoefffn
        self.times = times
        self.initial = initial
        self.noise_scale = noise_scale #should be 1 by default. allows you to forcibly turn up/down the noise during sampling. TODO: make this a proper schedule?
        
        super().__init__(to_arr,from_arr,conditioning,method,methodargs)
    
    def __len__(self) -> int:
        return len(self.times)

    def __iter__(self) -> Iterator[tuple[X, float]]:
        x = self.initial
        time_iter = iter(self.times)
        time_step = next(time_iter)
        for t2 in time_iter:
            yield (x,time_step)
            dt = (t2 - time_step) #note that this means dt is negative!!! so we've gotta negate it when we use it
            assert dt > 0, "dt must be positive for forward SDE sampling! Make sure you use a time schedule which increases monotonically"
            g = self.diffcoefffn(time_step)

            #dx = g(t)dw
            with torch.no_grad():
                x_arr = self.to_arr(x)
                dx = self.noise_scale * torch.sqrt(dt) * g * torch.randn_like(x_arr) 
                x_arr = x_arr + dx #don't do it in place! these tensors don't get cloned. (maybe they should?)
                x = self.from_arr(x_arr)

            time_step = t2

        yield (x,time_step)




#just pass in a sequence of points. time schedule is interpreted to be tmin..tmax linearly
class UniformIntegrableSequence(IntegrableSequence[X,C]):
    def __init__(self,points:Iterable[X], to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any],tmin:float=0, tmax:float=1):
        points = list(points)
        t = np.linspace(tmin,tmax,len(points),endpoint=True)
        super().__init__(list(zip(points,t)),to_arr,from_arr,conditioning,method,methodargs)

class InterpolatedUniformIntegrableSequence(InterpolatedIntegrableSequence[X,C]):
    def __init__(self,points:Iterable[X], n_interp: int,  to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any],tmin:float=0, tmax:float=1):
        points = list(points)
        t = np.linspace(tmin,tmax,len(points),endpoint=True)
        super().__init__(n_interp,list(zip(points,t)),to_arr,from_arr,conditioning,method,methodargs)


# Piecewise Differentiable Sequence. Note that since dx and dt are not defined at the vertices, it's not ODEIntegrable. However, since the subpaths are LinearPaths, they can be odeintegrated
# individually. Thus, by default this path will use ode integration using the provided 'method'. However, this behavior can be changed using the 'integral_type' method argument.
# Additionally, to forgo the custom diffintegrate method, use {"integral_type":"original"} for the method arguments. The method and all other method arguments will be used in the default
# diffintegrate instead of the subpaths'.

#TODO: Lots of possible generalizations. Different subpaths other than LinearPath, ways to define other than sequence, etc
class PiecewiseDifferentiableSequence(IntegrablePath[X,C]):
    def __init__(self,points:Iterable[tuple[X,float]],n_interp:int, to_arr:Callable[[X],Tensor], from_arr:Callable[[ArrayLike],X], conditioning:C, method:str, methodargs:dict[str,Any]):
        
        if not isinstance(points,Sequence):
            points = list(points)
        self.points = points

        super().__init__(to_arr,from_arr,conditioning,method,methodargs)
        methodargs = methodargs.copy()
        self.integral_type = methodargs.pop("integral_type","ode") #remove integral_type so it doesn't interfere with LinearPath's method args

        # Like InterpolatedIntegrableSequence, adds n_interp - 1 points to each step. That is, each step (x1,x2) will go from having 2 points to n_interp + 1 points (sharing endpoints with neighboring steps).
        # Note that while each path contains the entire n_interp - 1 points for integration purposes, iterating over this path will skip the first point of each path except the initial one so no points are repeated
        assert n_interp >= 1
        self.n_interp = n_interp
        
        self.pathclass = LinearPath #for forward compatibility
        self.paths = [
            LinearPath(x1,x2,np.linspace(0,1,n_interp+1),to_arr,from_arr,conditioning,method,methodargs) for (x1,x2) in itertools.pairwise(points)
        ]

        if self.integral_type == 'ode' and not issubclass(self.pathclass,ODEIntegrablePath):
            raise ValueError(f"Can't use ODE integration for non-integrable piecewise path: {self.pathclass}")

        

    @override
    def diffintegrate(self, *integrands: LikelihoodIntegrand[X, C], accumulate=True) -> tuple[Sequence[X], Sequence[float], list[list[float | Tensor]]]:
        if self.integral_type == 'original':
            return super().diffintegrate(*integrands)
        elif self.integral_type == 'diff' or self.integral_type == 'ode':
            x,t = self.points[0]
            likelihoods = [i.zero(x) for i in integrands]
            
            accx = [x] if accumulate else []
            acct = [t] if accumulate else []
            acc:list[list[float|Tensor]] = [[l] for l in likelihoods] if accumulate else []

            for path in self.paths:
                #get subpath likelihoods
                if self.integral_type == 'diff':
                    X,T,L = path.diffintegrate(*integrands, accumulate=accumulate)
                elif self.integral_type == 'ode':
                    assert isinstance(path,ODEIntegrablePath)
                    X,T,L = path.odeintegrate(*integrands, accumulate=accumulate)
                else:
                    raise Exception()

                #add subpath likelihoods to running total (and accumulate trajectory if specified)
                x = X[-1]
                t = T[-1]
                if accumulate:
                    accx.extend(X)
                    acct.extend(T)
                    [accl.extend([l0 + li for li in Li]) for accl,Li,l0 in zip(acc,L,likelihoods)] #add the current accumulated likelihood value to each of the likelihoods returned by the subpath
                    likelihoods = [accl[-1] for accl in acc] #the final step of each of the accumulated likelihoods
                else:
                    likelihoods = [l0 + Li[-1] for Li,l0 in zip(L,likelihoods)]

            if accumulate:
                return accx,acct,acc
            else:
                return [x],[t],[[l] for l in likelihoods]

        else:
            raise ValueError(f"Unknown integral type: {self.integral_type}")

    def __len__(self) -> int:
        return len(self.paths)*(self.n_interp-1) + 1

    def __iter__(self) -> Iterator[tuple[X, float]]:
        for i,path in enumerate(self.paths):
            for j,point in enumerate(path):
                if j != 0 and i == 0: #skip the first point of every path except the first to prevent endpoint overlap
                    yield point


#this is kind of strecthing the definition of an integrable 'path' but whatever. The "right" way would be to add a *more* abstract class like
#AbstractIntegrableObject which just needs a diffintegrate method but I'm not doing that lol.
class EnsembledIntegrablePath(IntegrablePath[X,C]):
    def __init__(self, paths:Iterable[IntegrablePath[X,C]], to_arr: Callable[[X], Tensor], from_arr: Callable[[ArrayLike], X], conditioning: C, method: str, methodargs: dict[str, Any]):
        #I don't think method or methodargs will be used here but more arguments are always nice. They could hold weighted average information or sometihng
        super().__init__(to_arr, from_arr, conditioning, method, methodargs)

        #at the moment this class is very dumb, just integrating paths in parallel (then averaging) with no understanding of what they are.
        #We might later find that it's better to average based on the path (e.g. marginalization by multiplying by relative probability of that path or something)
        self.paths = paths 

    def diffintegrate(self, *integrands: LikelihoodIntegrand[X, C], accumulate=True) -> tuple[Sequence[X], Sequence[float], list[list[float | Tensor]]]:
        xres = []
        tres = []
        lres = [[]]*len(integrands)

        for path in self.paths:
            X,T,L = path.diffintegrate(*integrands,accumulate=accumulate)
            xres.append(X)
            tres.append(T)
            [lres[i].append(L[i]) for i in range(len(L))]
        
        xres = [torch.stack(x,dim=0) for x in itertools.zip_longest(*xres,fillvalue=torch.nan)]
        tres = [torch.stack(t,dim=0) for t in itertools.zip_longest(*tres,fillvalue=torch.nan)]
        lres = [list(itertools.zip_longest(*res,fillvalue=torch.nan)) for res in lres]

        return xres,tres,lres #idk maybe it works
    
    #again this should maybe be just an integrable class not an integrable path but w/e
    def __iter__(self) -> Iterator[tuple[X, float]]:
        raise ValueError()
    
    def __len__(self) -> int:
        raise ValueError()

class UniformODEIntegrablePath(ODEIntegrablePath[X,C]):
    def dt(self, i: float, x: X, t: float) -> float:
        return 1

class FlowEquivalentODEPath(UniformODEIntegrablePath[X,C]):
    def __init__(self, scorefn:Callable[[X,float,C],Tensor], diffcoefffn:Callable[[float],float], timeschedule: Sequence[float], initial: tuple[X, float], to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any]):
        self.scorefn = scorefn
        self.diffcoefffn = diffcoefffn
        super().__init__(timeschedule, initial, to_arr, from_arr,conditioning,method,methodargs)

    def dx(self, i: float, x: X, t: float) -> X:
        delta = -1/2 * self.diffcoefffn(t)**2 * self.tensor(self.scorefn(x,t,self.condition)).detach()

        return self.from_arr(delta)

class LinearPath(ODEIntegrablePath[X,C]):
    def __init__(self, start:tuple[X,float], end:tuple[X,float], interpolants:Sequence[float], to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any]):
        super().__init__(interpolants,start,to_arr,from_arr,conditioning,method,methodargs)
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
    def __init__(self, scorefn:Callable[[X,float,C],Tensor], diffcoefffn:Callable[[float],float], timeschedule: Sequence[float], initial: tuple[X, float], to_arr:Callable[[X],Tensor],from_arr:Callable[[ArrayLike],X],conditioning:C, method:str, methodargs:dict[str,Any],interpolants:Optional[Sequence[float]]=None):
        self.flowpath = FlowEquivalentODEPath(scorefn,diffcoefffn,timeschedule,initial,to_arr,from_arr,conditioning,method,methodargs)
        self.path = list(self.flowpath)
        end = self.path[-1]
        super().__init__(initial,end,interpolants or timeschedule,to_arr,from_arr,conditioning,method,methodargs)



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
    def __init__(self, path: IntegrablePath[X,C], perturbation_schedule:Literal["uniform","data"], sigma: float, conditioning:C, method:str, methodargs:dict[str,Any]):
        points = list(path)
        shape = path.to_arr(points[0][0]).shape
        if perturbation_schedule == "data":
            times = [torch.as_tensor(point[1]).item() for point in points]
        elif perturbation_schedule == "uniform":
            times = np.linspace(0,1,len(points))
        else:
            raise ValueError(perturbation_schedule)
        offsets = brownian_bridge(times, shape, sigma=sigma)

        super().__init__([(x + offset, t) for (x,t),offset in zip(points,offsets)],path.to_arr,path.from_arr,conditioning,method,methodargs)








#### Likelihood Calculation ####


## I hate this so fucking much. I'm going to write a manifesto and send it to pytorch in the mail
# from https://discuss.pytorch.org/t/best-way-to-convert-a-list-to-a-tensor/59949/8
def tensorify(lst,device=None,dtype=None):
    """
    List must be nested list of tensors (with no varying lengths within a dimension).
    Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

    :return: nested list D
    """
    # base case, if the current list is not nested anymore, make it into tensor
    if type(lst) is torch.Tensor: return lst
    if type(lst[0]) is not list:
        if type(lst) is torch.Tensor:
            return lst
        elif type(lst[0]) is torch.Tensor:
            return torch.stack(lst, dim=0)
        else:  # if the elements of lst are floats or something like that
            try:
                return torch.as_tensor(lst,dtype=dtype,device=device)
            except ValueError:
                print(lst[0])
                raise 
    current_dimension_i = len(lst)
    for d_i in range(current_dimension_i):
        tensor = tensorify(lst[d_i])
        lst[d_i] = tensor
    # end of loop lst[d_i] = tensor([D_i, ... D_0])
    tensor_lst = torch.stack(lst, dim=0)
    return tensor_lst

_I = TypeVar("_I") # id type
def _run_likelihood(method:Literal['diff','ode'],path:IntegrablePath[X,C],integrands:Sequence[LikelihoodIntegrand[X,C]],accumulate:bool=True)->tuple[Sequence[X],Sequence[float],dict[str,list[np.ndarray]]]:

    with torch.profiler.record_function("Likelihood Integration"):
        if method == 'diff':
            trajectory, times, deltas = path.diffintegrate(*integrands,accumulate=accumulate)
        elif method == 'ode':
            if not isinstance(path,ODEIntegrablePath):
                raise ValueError(f"Path {path} is not ODEIntegrable! Please use an ODEIntegrable path or set the integral_type to 'diff' to use the path in euclidean mode")
            trajectory, times, deltas = path.odeintegrate(*integrands,accumulate=accumulate)
    ##Since we assume the path goes from unknown to known, we negate the delta. The last data point is the accumulated integrand (but we pass the whole thing as output so we can save it)
    integrand_results:dict[str,list[np.ndarray]] = {integrand.name(): ([-tensorify(delta[i],device='cpu').detach().cpu().numpy() for i in range(len(delta))]) for integrand,delta in zip(integrands,deltas)}

    return (trajectory, times, integrand_results)

def run_diff_likelihood(path:IntegrablePath[X,C],integrands:Sequence[LikelihoodIntegrand[X,C]],accumulate:bool=True):
    return _run_likelihood('diff',path,integrands,accumulate=accumulate)

def run_ode_likelihood(path:ODEIntegrablePath[X,C],integrands:Sequence[ODELikelihoodIntegrand[X,C]],accumulate:bool=True):
    return _run_likelihood('ode',path,integrands,accumulate=accumulate)
