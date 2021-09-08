x = np.random.randn(3,5000)
xnormed = x/np.linalg.norm(x, axis=0)

plt.clf()
ax = plt.axes(projection ='3d')
w = np.random.rand(20)
#w = w/np.sum(w)
xx = xnormed.copy()*0
for i in range(20):
    kappa = np.random.randint(1, 200)
    mu = np.random.randn(3); mu = mu/np.linalg.norm(mu)
    y = apply_vmf(xnormed, mu, kappa)
    xx += w[i]*y
ax.scatter(xx[0,:], xx[1,:], xx[2,:], c=np.linalg.norm(xx,axis=0)-1, s=30, cmap=plt.cm.inferno)
#plt.gca().set_aspect(1)
#plt.axis('off')
plt.show()

##################3
x = np.random.randn(3,5000)
xnormed = x/np.linalg.norm(x, axis=0)

plt.clf()
ax = plt.axes(projection ='3d')
w = np.random.rand(20)
#w = w/np.sum(w)
xx = xnormed.copy()*0
for i in range(20):
    kappa = np.random.randint(1, 200)
    mu = np.random.randn(3); mu = mu/np.linalg.norm(mu)
    y = apply_vmf(xnormed, mu, kappa)
    xx += w[i]*y
ax.scatter(xx[0,:], xx[1,:], xx[2,:], c=np.linalg.norm(xx,axis=0)-1, s=30, cmap=plt.cm.inferno)
#plt.gca().set_aspect(1)
#plt.axis('off')
plt.show()
###################
import numpy as np

x = np.random.randn(2,5000)
xnormed = x/np.linalg.norm(x, axis=0)

np.save(xnormed, 'only_use_these_points_for_coordinates_never_regenerate.npy')
