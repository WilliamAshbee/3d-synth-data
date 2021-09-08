from scipy.special import iv as besseli
import pylab as plt
import numpy as np

xbaseline = np.random.randn(2, 1000)


def vmf(mu, kappa, x):
    # single point function
    d = mu.shape[0]
    # compute in the log space
    logvmf = (d // 2 - 1) * np.log(kappa) - np.log(
        (2 * np.pi) ** (d / 2) * besseli(d // 2 - 1, kappa)) + kappa * np.dot(mu, x)
    return np.exp(logvmf)


def apply_vmf(x, mu, kappa, norm=1.0):
    delta = 1.0 + vmf(mu, kappa, x)
    y = x * np.c_[delta, delta].T
    return y


def createDataset(pltFig=True, numFigs=100, numPoints=1000):
    dimensions = 2
    # x = np.random.randn(numFigs,dimensions, numPoints)
    ##create an ordered circle instead of a permuted circle
    x = np.zeros((numFigs, dimensions,
                  numPoints))  # this could get interesting! why does it fail on the random permuted version
    xnormed = xbaseline
    xnormed = xnormed / np.linalg.norm(xnormed, axis=0)
    x[:, :, :] = xnormed
    # theta = np.linspace(0, 2.0*3.14, num=1000)
    # xx = np.cos(theta)
    # yy = np.sin(theta)
    # x[:,0,:]= xx
    # x[:,1,:]= yy
    # assert np.sum(x[0,0,:]!=x[1,0,:])==0
    # assert np.sum(x[0,1,:]!=x[1,1,:])==0

    for k in range(numFigs):
        # plt.clf()
        if pltFig:
            fig = plt.figure()
        w = np.random.rand(11)
        w = w / np.sum(w)
        # x = np.random.randn(dimensions,numpoints)
        xnormed = x[k, :, :]  # /np.linalg.norm(x[k,:,:], axis=0)
        xx = x[k, :, :] * w[0]
        it = 10
        for i in range(it):
            kappa = np.random.randint(5, 100)
            mu = np.random.randn(2);
            mu = mu / np.linalg.norm(mu)
            y = apply_vmf(xnormed, mu, kappa)
            xx += w[i + 1] * y
        x[k, :, :] = xx
        if pltFig:
            plt.plot(x[k, 0, :], x[k, 1, :], ',-', ms=1)
            print(xx.shape)
            plt.gca().set_aspect(1)
            # plt.axis('off')
            plt.show()

    return x


x = createDataset(pltFig=True)
