from matplotlib.colors import LogNorm
import numpy as np
import torch
from diffenergy.groundtruth_score import batched_normpdf_scalar
x = np.linspace(-20,20,1000)
y = np.linspace(-20,20,1000)
Y, X = np.meshgrid(y,x)

XY = np.stack([X,Y],axis=-1).reshape((-1,2))

N = 10

mu = np.array([[ 12.59776771,   1.70518727],
       [ -1.98126094, -11.72397409],
       [ 14.16377112,  13.93838231],
       [  1.69974902, -12.97185663],
       [  4.76352451,  14.56532704],
       [  2.61977162,  13.96935809],
       [  7.40388344,   9.66828433],
       [-11.3459567 ,  -1.19982939],
       [ 11.31086002, -12.27363392],
       [ -8.30775642, -11.82986102]])
sigma = np.array([12, 10, 11, 16, 11, 12,  9,  8,  8, 11])+250
weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# mu = np.random.uniform(-15,15,size=(N,2))
# sigma = np.abs(np.random.poisson(12,size=(N,)))
# weights = np.ones((N,),dtype=float)/N

dx = XY[:,None] - mu[None,:]

print(dx.shape)

print(dx.min())

pdf = batched_normpdf_scalar(torch.as_tensor(dx),torch.as_tensor(sigma))

print(pdf.min())

pdf = pdf.numpy().reshape((1000,1000,N))

pdf = pdf.sum(axis=-1)

from matplotlib import cm, rcParams
rcParams["axes3d.mouserotationstyle"] = "azel"
import matplotlib.pyplot as plt

fig,ax = plt.subplots(subplot_kw={"projection":"3d"})

Y, X = np.meshgrid(y,x)

ax.plot_surface(X,Y,pdf, rstride = 10, cstride = 10, cmap = "jet")
ax.set_zlabel("probability density")
plt.show()

from IPython import embed; embed()