from scipy.special import iv as besseli
import numpy as np
import matplotlib.pyplot as plt
import icosahedron as ico#local file
from datetime import datetime


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


from sklearn.metrics.pairwise import cosine_similarity


def dedup(mat):
    mat = mat - np.mean(mat, axis=1).reshape(3, 1)
    # datetime object containing current date and time
    now = datetime.now()
    start = now.strftime("%d/%m/%Y %H:%M:%S")
    mat = (mat * (1.0 / np.linalg.norm(mat, axis=0).reshape(1, -1)))
    similarities = cosine_similarity(mat.T)
    similarities = similarities >.99999
    similarities = np.triu(similarities)  # upper triangular
    np.fill_diagonal(similarities, 0)  # fill diagonal
    similarities = np.sum(similarities, axis=0)
    similarities = similarities == 0  # keep values that are no one's duplicates
    mat = mat[:, similarities]
    now = datetime.now()
    end = now.strftime("%d/%m/%Y %H:%M:%S")
    return mat

global firstDone
firstDone = False
global x
x = ico.icosphere(30, 1.3)
ind = (x[0, :] ** 2 + x[1, :] ** 2 + x[2, :] ** 2) >= (
            np.median((x[0, :] ** 2 + x[1, :] ** 2 + x[2, :] ** 2)) - .0001)  # remove zero points
x = x[:, ind]  # fix to having zero values
x = dedup(x)


def createOneMutatedIcosphere():
    global firstDone
    global x
    numbumps = 50
    w = np.random.rand(numbumps)
    w = w/np.sum(w)


    #xnormed = x/np.linalg.norm(x, axis=0)
    xnormed = x#norming in dedup now
    xx = np.zeros_like(xnormed)

    for i in range(numbumps):
        kappa = np.random.randint(1, 200)
        mu = np.random.randn(3); mu = mu/np.linalg.norm(mu)
        y = apply_vmf(xnormed, mu, kappa)
        xx += w[i]*y

    return xx

mutIcos = np.zeros((1,3,9002))
maxs = 0
mins = 1000000

np.random.seed(0)
for i in range(mutIcos.shape[0]):

    mutIco = createOneMutatedIcosphere()
    mutIcos[i,:,:] = mutIco
#    if i%200 == 0:
#        print(maxs, mins)
#    maxs = np.max([maxs,mutIco.shape[1]])
#    mins = np.min([mins, mutIco.shape[1]])



