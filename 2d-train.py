from scipy.special import spherical_jn as besseli
import numpy as np
from scipy import *
from math import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import pylab as plt
#from skimage import filters
import math

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from scipy.special import iv as besseli
import pylab as plt
import numpy as np
from numpy.core.function_base import linspace
from numpy.linalg import linalg

xbaseline = np.random.randn(2, 1000)

useGPU = False
mini_batch = 100
train_size = mini_batch*1

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




global numpoints
numpoints = 1000
side = 32

rows = torch.zeros(32, 32)
columns = torch.zeros(32, 32)

for i in range(32):
    columns[:, i] = i
    rows[i, :] = i


def convex_combination_matrix(length=10):
    canvas = torch.zeros((length, side, side))
    points = createDataset(pltFig=False, numFigs=length, numPoints=1000).transpose(0, 2, 1)
    maxP = np.max(points)
    minP = np.min(points)
    points = (points - minP) * 26.0 / (maxP - minP) + 3.0
    points = torch.from_numpy(points)

    assert points.shape == (length, 1000, 2)

    for l in range(length):
        canvas[l, points[l, :, 0].type(torch.LongTensor), points[l, :, 1].type(torch.LongTensor)] = 1.0

        # points[l,:,0] = x[l,:]#modified for lstm discriminator
        # points[l,:,1] = y[l,:]#modified for lstm discriminator

    print('canvshape', canvas.shape)
    print('sumcanvas', torch.sum(canvas) / length)

    return {
        'canvas': canvas,
        'points': points}


def plot_one(img, xs, ys, i=0):
    print(img.shape, xs.shape, ys.shape)
    plt.subplot(10, 10, i + 1)
    # print(type(img))
    # print(np.max(img))
    plt.imshow(img.T, cmap=plt.cm.gray_r)
    predres = 1000
    s = [.001 for x in range(predres)]
    assert len(s) == predres
    c = ['red' for x in range(predres)]
    assert len(c) == predres
    plt.plot(xs.cpu().numpy(), ys.cpu().numpy(), ',', color='red', ms=.3, lw=.3)
    # plt.gca().add_artist(ascatter)
    plt.axis('off')


def plot_all(sample=None, model=None, labels=None, i=0):
    # make prediction
    # plot one prediction
    # or plot one ground truth
    if model != None:
        with torch.no_grad():
            global numpoints

            print('preloss')
            if useGPU:
                loss, out = mse_vit(sample.cuda(), labels.cuda(), model=model, ret_out=True)
            else:
                loss, out = mse_vit(sample, labels, model=model, ret_out=True)

            print('loss', loss)

            for i in range(100):
                img = sample[i, 0, :, :].squeeze().cpu().numpy()
                X = out[i, :1000, 0]
                Y = out[i, -1000:, 1]
                plot_one(img, X, Y, i=i)


    else:
        print(sample.shape, labels.shape)
        for i in range(100):
            img = sample[i, 0, :, :].squeeze().cpu().numpy()
            # print(np.sum('img',img))
            X = labels[i, :, 0]
            Y = labels[i, :, 1]
            # print('x',np.max(X,dim=None),np.min(X,dim=None))
            # print('y',np.max(Y,dim=None),np.min(Y,dim=None))
            plot_one(img, X, Y, i=i)


class DonutDataset(torch.utils.data.Dataset):
    """Donut dataset."""

    def __init__(self, length=10):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.length = length
        self.values = convex_combination_matrix(length)
        assert self.values['canvas'].shape[0] == self.length
        assert self.values['points'].shape[0] == self.length


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        canvas = self.values["canvas"]

        canvas = canvas[idx, :, :]

        # assert canvas.shape == (side,side) or canvas.shape == (len(idx))

        points = self.values["points"]
        points = points[idx, :]
        if len(canvas.shape) == 2:
            canvas = torch.stack([canvas, canvas, canvas], dim=0)
        else:
            canvas = torch.stack([canvas, canvas, canvas], dim=1)

        return canvas, points

    @staticmethod
    def displayCanvas(title, loader, model):
        # model.setBatchSize(batch_size = 1)

        for sample, labels in loader:
            plot_all(sample=sample, model=model, labels=labels)
            break
        plt.savefig(title, dpi=600)


dataset = DonutDataset(length=100)

loader_demo = data.DataLoader(
    dataset,
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset),
    num_workers=2)
DonutDataset.displayCanvas('donut.png', loader_demo, model=None)


from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

mini_batch = 100
train_dataset = DonutDataset(length = train_size)
loader_train = data.DataLoader(
    train_dataset,
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=train_dataset),
    num_workers=4)


import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 32,
    patch_size = 8,
    num_classes = 2000,
    dim = 2048,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

v = torch.nn.Sequential(
    v,
    torch.nn.Sigmoid()
)

img = torch.randn(100, 3, 32, 32)

preds = v(img) # (1, 1000)

if useGPU:
    model = v.cuda()
else:
    model = v
def mse_vit(input, target,model=None,ret_out = False):
  out = model(input)
  #print('out,target',out.shape,target.shape)
  out = out.reshape(target.shape)#64, 1000, 2
  out = out*32.0
  #print('mse_cnn',out.shape,target.shape)
  if not ret_out:
    return torch.mean((out-target)**2)
  else:
    return torch.mean((out-target)**2),out

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001, betas = (.9,.999))#ideal


for epoch in range(1):
  for x,y in loader_train:
    optimizer.zero_grad()
    if useGPU:
        x = x.cuda()
        y = y.cuda()
    #print('x,y',x.shape,y.shape)
    loss = mse_vit(x,y,model=model)
    loss.backward()
    optimizer.step()
  print('epoch',epoch,'loss',loss)
    #print(out.shape)

model = model.eval()
#dataset = DonutDataset(length = 100)
DonutDataset.displayCanvas('vit-training-1.png',loader_train, model = model)


from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

mini_batch = 100
dataset = DonutDataset(length = 100)
loader_test = data.DataLoader(
    dataset,
    batch_size=mini_batch,
    sampler=RandomSampler(data_source=dataset),
    num_workers=4)

for x,y in loader_test:
  if useGPU:
      x = x.cuda()
      y = y.cuda()
  loss = mse_vit(x,y,model=model)
  print('validation loss',loss)
  break

DonutDataset.displayCanvas('vit-test-set-1.png',loader_test, model = model)

