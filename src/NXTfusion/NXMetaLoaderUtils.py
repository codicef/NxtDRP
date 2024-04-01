import pickle as cPickle
import time
import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class MetaLoader:

	def __init__(self, loaders):
		self.loaders = loaders

	def __iter__(self):
		return MetaLoaderIter(self)

class MetaLoaderIter:

	def __init__(self, metaLoader):
		self.loadersIter = self.createIterators(metaLoader.loaders)
		self.counter = []
	
	def createIterators(self, loaders):
		loadersIterators = []
		i = 0
		while i < len(loaders):
			loadersIterators.append(iter(loaders[i]))
			i+=1
		return loadersIterators

	def __iter__(self):
		return self

	def __next__(self):
		r = []
		for i in self.loadersIter:
			r.append(next(i))
		return r

	next = __next__


