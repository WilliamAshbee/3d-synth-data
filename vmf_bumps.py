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



ax.scatter(xx[0,:], xx[1,:], xx[2,:], c=np.linalg.norm(xx,axis=0)-1, s=0.5, cmap=plt.cm.inferno)
#for i in range(xx.shape[0]):
#    for j in range(xx.shape[0]):
#        print(np.cos(xx[i,:],xx[j,:]))

#print(np.dot(xx,xx.T))

#cos_sim = np.dot(xx.T, xx)/(np.linalg.norm(xx.T)*np.linalg.norm(xx))

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
start = now.strftime("%d/%m/%Y %H:%M:%S")

print('start', start)


from sklearn.metrics.pairwise import cosine_similarity
#from scipy import sparse

#count = 0
#for i in range(xx.shape[1]):
#    j = i+1
#    while j < xx.shape[1]:
 #       if np.dot(xx.T[i, :], xx[:, j]) / (np.linalg.norm(xx.T[i, :]) * np.linalg.norm(xx[:, j])) >1.0-.0001:
#            print('ij',i,j)
 #           count+=1
 #       j+=1
#
similarities = cosine_similarity(xx.T)

similarities = similarities>.999999

similarities = np.triu(similarities)#upper triangular
np.fill_diagonal(similarities,0)#fill diagonal

similarities = np.sum(similarities,axis=0)

similarities = similarities == 0

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
