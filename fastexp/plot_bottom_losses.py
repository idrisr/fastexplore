from fastai.data.all import *
from fastai.vision.all import *

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(path)
learner = cnn_learner(dls, resnet18)
