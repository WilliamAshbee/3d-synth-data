from scipy.special import iv as besseli
import pylab as plt
import numpy as np
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import icosahedron as ico#local file


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
ind = (x[0,:]**2+x[1,:]**2+x[2,:]**2)>=(np.median((x[0,:]**2+x[1,:]**2+x[2,:]**2))-.0001)
x = x[:,ind] #fix to having zero values
xnormed = x/np.linalg.norm(x, axis=0)
xx = np.zeros_like(xnormed)

for i in range(numbumps):
    kappa = np.random.randint(1, 200)
    mu = np.random.randn(3); mu = mu/np.linalg.norm(mu)
    y = apply_vmf(xnormed, mu, kappa)
    xx += w[i]*y

xx = xx-np.mean(xx,axis=1).reshape(3,1)
print(np.mean(xx,axis=1),'mean')

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
start = now.strftime("%d/%m/%Y %H:%M:%S")

print('start', start)


from sklearn.metrics.pairwise import cosine_similarity

xxtemp = (xx*(1.0/np.linalg.norm(xx,axis=0).reshape(1,-1)))
#ax.scatter(xxtemp[0,:], xxtemp[1,:], xxtemp[2,:], c=np.linalg.norm(xxtemp,axis=0)-1, s=0.5, cmap=plt.cm.inferno)

similarities = cosine_similarity(xxtemp.T)

similarities = similarities>.999999

similarities = np.triu(similarities)#upper triangular
np.fill_diagonal(similarities,0)#fill diagonal

similarities = np.sum(similarities,axis=0)

similarities = similarities == 0 #keep values that are no one's duplicates

print('keep' , np.sum(similarities))
xx = xx[:,similarities]
print(xx.shape)
now = datetime.now()
end = now.strftime("%d/%m/%Y %H:%M:%S")

print('start', start)

print('end', end)

#plt.gca().set_aspect(1)
#plt.axis('off')
#plt.show()
ax.scatter(xx[0,:], xx[1,:], xx[2,:], c=np.linalg.norm(xx,axis=0)-1, s=0.5, cmap=plt.cm.inferno)
plt.show()