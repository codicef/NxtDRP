import pickle as cPickle
import time
import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def predictCollate(batch):
	x1 = []
	x2 = []		
	sx1 = []
	sx2 = []
	for i in  batch:		
		x1.append(i[0])
		x2.append(i[1])
		if len(i) > 2:
			sx1.append(i[2])
			sx2.append(i[3])
	if len(sx1) > 0 or len(sx2) > 0:	
		#print sx1[0]
		return t.LongTensor(x1), t.LongTensor(x2), t.FloatTensor(np.vstack(sx1)), t.FloatTensor(np.vstack(sx2))
	else:
		return t.LongTensor(x1), t.LongTensor(x2)

def predictMetaCollate(batch):
	x1 = []
	x2 = []
	side1 = []
	side2 = []
	for i in  batch:
		#assert i[0].shape == i[1].shape == i[2].shape
		x1.append(i[0])
		x2.append(i[1])
		if len(batch[0]) > 3:
			if len(i[2]) > 0:
				side1.append(i[2])
			if len(i[3]) > 0:
				side2.append(i[3])
	#print x1
	#print y
	x1 = np.hstack(x1)
	x2 = np.hstack(x2)
	#print len(side1), type(side1)
	if len(batch[0]) > 3:
		if len(side1) == 0:
			side1 = None
		else:
			side1 = np.vstack(side1)
			#print side1.size()
			#assert side1.size()[0] == len(x1)
		if len(side2) == 0:
			side2 = None
		else:
			tmp1 = np.vstack(side2)
			#print tmp1.shape
			side2 = tmp1
			#print "qui",side2.size()
			#assert side2.size()[0] == len(x2)
		return x1, x2, side1, side2
	else:
		return x1, x2

def metaCollate(batch):
	x1 = []
	x2 = []
	y = []
	side1 = []
	side2 = []
	for i in  batch:
		#assert i[0].shape == i[1].shape == i[2].shape
		x1.append(i[0])
		x2.append(i[1])
		y.append(i[2])
		if len(batch[0]) > 3:
			if len(i[3]) > 0:
				side1.append(i[3])
			if len(i[4]) > 0:
				side2.append(i[4])
	#print x1
	#print y
	x1 = np.hstack(x1)
	x2 = np.hstack(x2)
	y = np.vstack(y)
	#print len(side1), type(side1)
	if len(batch[0]) > 3:
		if len(side1) == 0:
			side1 = None
		else:
			side1 = np.vstack(side1)
			#print side1.size()
			#assert side1.size()[0] == len(x1)
		if len(side2) == 0:
			side2 = None
		else:
			side2 = np.vstack(side2)
			#print tmp1.shape
			#side2 = t.FloatTensor(tmp1).squeeze()
			#print "qui",side2.size()
			#assert side2.size()[0] == len(x2)
		return x1, x2, y, side1, side2
	else:
		return x1, x2, y

