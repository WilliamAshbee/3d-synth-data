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
"""
for i in range(mutIcos.shape[0]):

    mutIco = createOneMutatedIcosphere()
    mutIcos[i,:,:] = mutIco
"""
#    if i%200 == 0:
#        print(maxs, mins)
#    maxs = np.max([maxs,mutIco.shape[1]])
#    mins = np.min([mins, mutIco.shape[1]])


import torch
import numpy as np
import pylab as plt
#from skimage import filters
import math

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

global numpoints
numpoints = 9002
side = 8
sf = .99999
xs = np.zeros((side,side,side))
ys = np.zeros((side,side,side))
zs = np.zeros((side,side,side))

for i in range(side):
    xs[i,:,:] = i
    ys[:,i,:] = i
    zs[:,:,i] = i

def rasterToXYZ(r):#may need to be between 0 and 7 instead of 0 and side*sf
    a = np.copy(r)
    xr = (xs * a)[r == 1]

    yr = (ys * a)[r == 1]

    zr = (zs * a)[r == 1]

    xr = side*sf*(xr - np.min(xr)) * (1.0 / (np.max(xr) - np.min(xr)))
    yr = side*sf*(yr - np.min(yr)) * (1.0 / (np.max(yr) - np.min(yr)))
    zr = side*sf*(zr - np.min(zr)) * (1.0 / (np.max(zr) - np.min(zr)))

    return xr,yr,zr

def mutated_icosphere_matrix(length=10,canvas_dim=8):
    points = torch.zeros(length, numpoints, 3).type(torch.FloatTensor)
    canvas = torch.zeros(length,canvas_dim,canvas_dim,canvas_dim).type(torch.FloatTensor)


    for l in range(length):
        xx = createOneMutatedIcosphere()
        xx = (xx - np.expand_dims(np.min(xx, axis=1), axis=1)) * np.expand_dims(
            1.0 / (np.max(xx, axis=1) - np.min(xx, axis=1)), axis=1)
        xx = torch.from_numpy(xx)
        xx = xx*sf
        print(xx.shape)
        x = xx[0,:]
        print(x.shape)
        y = xx[1,:]
        z = xx[2,:]

        points[l, :, 0] = x[:]  # modified for lstm discriminator
        points[l, :, 1] = y[:]  # modified for lstm discriminator
        points[l, :, 2] = z[:]  # modified for lstm discriminator

        canvas[l, (x*side).type(torch.LongTensor), (y*side).type(torch.LongTensor), (z*side).type(torch.LongTensor)] = 1.0


    return {
        'canvas': canvas,
        'points': points.type(torch.FloatTensor)}


def plot_one(fig,img, xx, i=0):
    #print(img.shape, xs.shape, ys.shape)
    print(xx.shape)
    #plt.subplot(10, 10, i + 1,projection='3d')
    #plt.imshow(img, cmap=plt.cm.gray_r)

    predres = numpoints
    s = [.001 for x in range(predres)]
    assert len(s) == predres
    c = ['red' for x in range(predres)]


    s = [.01 for x in range(predres)]
    assert len(s) == 9002
    assert len(c) == predres
    #ax = plt.axes(projection='3d')
    ax = fig.add_subplot(10, 10, i + 1,projection='3d')
    ax.set_axis_off()
    #ax.set_xticklabels([])

    #ax.set_yticklabels([])

    ax.scatter(xx[:, 0]*side*sf, xx[:, 1]*side*sf,xx[:, 2]*side*sf, s=.01, c='red',cmap=plt.cm.inferno)
    #xx = xx+.5
    gtx,gty,gtz = rasterToXYZ(img)
    print('gt size',gtx.shape,gty.shape,gtz.shape)
    ax.scatter(gtx, gty, gtz, s=.01, c='black')

    print('begin')
    print('xxshape', xx.shape)
    print('tempmax', torch.max(xx[:, 0]))

    #print('shapes',np.max(xx[:, 0]).shape , np.max(gtx).shape)
    gtx = torch.from_numpy(gtx)
    gty = torch.from_numpy(gty)
    gtz = torch.from_numpy(gtz)

    print('maxes')
    print("xx from points")
    print(torch.max(xx[:, 0]), torch.max(xx[:, 1]), torch.max(xx[:, 2]))
    print(torch.min(xx[:, 0]), torch.min(xx[:, 1]), torch.min(xx[:, 2]))
    print('gtxyz from raster')
    print(torch.max(gtx), torch.max(gty), torch.max(gtz))
    print(torch.min(gtx), torch.min(gty), torch.min(gtz))





#plt.plot(xs.cpu().numpy(), ys.cpu().numpy(), ',-', color='red', ms=.3, lw=.3)
    # plt.gca().add_artist(ascatter)
    #plt.axis('off')


def plot_all(sample=None, model=None, labels=None, i=0):
    # make prediction
    # plot one prediction
    # or plot one ground truth
    if model != None:
        with torch.no_grad():
            global numpoints

            print('preloss')
            loss, out = mse_vit(sample.cuda(), labels.cuda(), model=model, ret_out=True)
            print('loss', loss)
            fig = plt.figure()
            for i in range(100):
                img = sample[i, 0, :, :].squeeze().cpu().numpy()
                X = out[i, :1000, 0]
                Y = out[i, -1000:, 1]

                plot_one(fig,img, X, Y, i=i)


    else:
        print(sample.shape, labels.shape)
        fig = plt.figure()
        for i in range(100):
            img = sample[i, :, :,:].squeeze().cpu().numpy()
            xx = labels[i, :]

            plot_one(fig,img, xx, i=i)


class MutatedIcospheresDataset(torch.utils.data.Dataset):
    """Donut dataset."""

    def __init__(self, length=10,canvas_dim = 8):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.length = length
        self.values = mutated_icosphere_matrix(length,canvas_dim)
        self.canvas_dim = canvas_dim
        assert self.values['canvas'].shape[0] == self.length
        assert self.values['points'].shape[0] == self.length

        count = 0
        for i in range(self.length):
            a = self[i]
            c = a[0][0, :, :]
            for el in a[1]:
                y, x = (int)(el[1]), (int)(el[0])

                if x < side - 2 and x > 2 and y < side - 2 and y > 2:
                    if c[y, x] != 1 and \
                            c[y + 1, x] != 1 and c[y + 1, -1 + x] != 1 and c[y + 1, 1 + x] != 1 and \
                            c[y - 1, x] != 1 and c[y, -1 + x] != 1 and c[y, 1 + x] != 1:
                        count += 1
        assert count == 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas = self.values["canvas"]

        canvas = canvas[idx, :, :]

        # assert canvas.shape == (side,side) or canvas.shape == (len(idx))

        points = self.values["points"]
        points = points[idx, :]
        #if len(canvas.shape) == 2:
        #    canvas = torch.stack([canvas, canvas, canvas], dim=0)
        #else:
        #    canvas = torch.stack([canvas, canvas, canvas], dim=1)

        return canvas, points

    @staticmethod
    def displayCanvas(title, loader, model):
        # model.setBatchSize(batch_size = 1)

        for sample, labels in loader:
            plot_all(sample=sample, model=model, labels=labels)
            break
        plt.savefig(title, dpi=600)


dataset = MutatedIcospheresDataset(length=100)

mini_batch = 100
loader_demo = data.DataLoader(
    dataset,
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset),
    num_workers=2)
MutatedIcospheresDataset.displayCanvas('mutatedicospheres.png', loader_demo, model=None)
