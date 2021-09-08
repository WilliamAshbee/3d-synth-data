from scipy.special import iv as besseli
import pylab as plt
import numpy as np
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import icosahedron as ico


def vmf(mu, kappa, x):
    # single point function
    d = mu.shape[0]
    # compute in the log space
    logvmf = (d//2-1) * np.log(kappa) - np.log((2*np.pi)**(d/2)*besseli(d//2-1,kappa)) + kappa * np.dot(mu,x)
    return np.exp(logvmf)

def apply_vmf(x, mu, kappa, norm=1.0):
    delta = 1.0+vmf(mu, kappa, x)
    y = x * np.vstack([np.power(delta,3)]*x.shape[0])
    return y

plt.clf()

ax = plt.axes(projection ='3d')
numbumps = 50
w = np.random.rand(numbumps)
w = w/np.sum(w)
x = ico.icosphere(30, 1.3) #np.random.randn(3,5000)
xnormed = x/np.linalg.norm(x, axis=0)
xx = xnormed.copy()*0
for i in range(numbumps):
    kappa = np.random.randint(1, 200)
    mu = np.random.randn(3); mu = mu/np.linalg.norm(mu)
    y = apply_vmf(xnormed, mu, kappa)
    xx += w[i]*y
ax.scatter(xx[0,:], xx[1,:], xx[2,:], c=np.linalg.norm(xx,axis=0)-1, s=0.5, cmap=plt.cm.inferno)
#plt.gca().set_aspect(1)
#plt.axis('off')
plt.show()
