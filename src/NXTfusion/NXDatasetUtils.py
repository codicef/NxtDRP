#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pytorchDatasetUtils.py
#  
#  Copyright 2018 Daniele Raimondi <daniele.raimondi@vub.be>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import pickle as cPickle
import time
import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class PredictionDatasetSide(Dataset):
	"""
	:meta private:
	"""
	def __init__(self, x, sx1, sx2):		
		#if sx1 != None and len(sx1) > 0:
		#	assert len(x) == len(sx1) 
		#if sx2 != None and len(sx2) > 0:
		#	assert len(sx2) == len(x)
		self.x = x # [(i,j), (i,k), ...]
		self.sx1 = sx1
		self.sx2 = sx2
				
	def __getitem__(self, idx): 
		"""
		:meta private:
		"""

		tmp = self.x[idx]
		sx1 = []
		sx2 = []
		if self.sx1 != None:
			sx1 = self.sx1[tmp[0]]
		if self.sx2 != None:
			sx2 = self.sx2[tmp[1]]
		return tmp[0], tmp[1], sx1, sx2		
	
	def __len__(self):
		return len(self.x)


class PredictionDataset(Dataset):
    
	def __init__(self, x, label = True):		
		self.x = x # [(i,j), (i,k), ...]
				
	def __getitem__(self, idx): 
		tmp = self.x[idx]
		return tmp[0], tmp[1]
		
	def __len__(self):
		return len(self.x)

class SideDataset(Dataset):
	def __init__(self, side):
		assert type(side) == {}
		self.side = side
		self.estSize = len(self) * len(self.values()[0])
	
	def __getitem__(self, idx):
		return self.side[idx]

	def __len__(self):
		return len(self.side)

class SubDataset(Dataset):
	"""
	Within the NNwrapper, during training, batches need to be rapidly provided for all the MetaRelations in the ERgraph and for each Relation in every MetaRelation. To do so, the NNwrapper.processDatasets function builds an internal Dataset structure that mimicks the structure of the input ERgraph. In this case, MetaDataset correspond to MetaRelation, and each Relation in a MetaRelation is represendet by a SubDataset in the corresponding MetaDataset.

	Nevertheless, this is internal and it is transparent to the user.
	:meta private:
	"""
	def __init__(self, xht, typep="binary"):		
		"""
		Constructor method for the SubDataset class. It puts in a pytorch-friendly structure the matrix corresponding to a target Relation, by transforming its DataMatrix into a pytorch Dataset.

		Parameters
		----------
		xht : dict
			Dict used to represent the matrix/relation data within a DataMatrix object
		type : str 
			String specifying the type of the prediction. It must be "regression" or "binary".

		Returns
		-------
		
		"""
		#print type(xht)
		assert type(xht) == dict
		self.xht = xht #xht = {p1:[(positions),(values)]}
		empty = 0
		print(f"Number of rows: {len(self.xht)}")
		for i in self.xht.items():
			# print (i[1][0])
			if len(i[1][0]) == 0:
				#print i[0]
				empty += 1
				#print i[1]
		
		print ("Empty rows: ",empty) 
		#raw_input()
		self.estSize = self.countInstances()
		self.type = typep	
		self.balance = self.countBalance()
		
	def countBalance(self):
		if self.type != "binary":
			return "regression"
			#raise Exception("CAlling count balance on regression problem!!!")
		r = [0,0]
		for i in self.xht.values():
			tmp = sum(i[1])
			r[0] += tmp
			r[1] += len(i[1])-tmp
		return r
		
	def countInstances(self):
		r = 0
		for i in self.xht.values():
			r += len(i[0])
		return r
		
	def __getitem__(self, idx):
		tmp = self.xht[idx]
		return tmp
		#return (np.array(tmp[0], dtype=np.int16), np.array(tmp[1], dtype=np.int8))
	
	@staticmethod
	def load(name):
		tmp = cPickle.load(open(name))
		vt = None
		if "binary" in name:
			vt = "binary"
		elif "regression" in name:
			vt = "regression"
		return SubDataset(tmp, typep=vt)
	
	def dump(self, name):
		print( "Dumping...")
		t1 = time.time()
		cPickle.dump(self.xht, open(name, "w"))
		t2 = time.time()
		print ("Stored in: %s (%.2fs)" % ( name, t2-t1))
		
	def __len__(self):
		return len(self.xht)		

class MetaDataset(Dataset): ###########cercato di ottimizzre lo spazio con numpy ma al collate arriva roba strana, da fixare !!!!!!!!!!#######################
	"""
	Class that represents the MetaRelations in the NNwrapper internal Dataset-based version of the ERgraph used for allowing a fast and consistent multi-task mini batching. Each MetaDataset can contain many SubDatasets, and when asked it provides a minibatch sampling from all of them in parallel.
	"""
	def __init__(self, datasetList, domain1, domain2, name, ignore_index, side1 = None, side2 = None):
		"""
		Constructor method for the MetaDataset class. It puts in a pytorch-friendly structure the data corresponding to a target MetaRelation, by storing several SubDataset (each corresponding to a Relation/DataMatrix/matrix).

		Parameters
		----------
		datasetList : list of SubDatasets
			List of Subdatasets. Each SubDataset corresponds to a Relation. The MetaDataset thus corresponds to a MetaRelation.
		domain1 : NX.Entity
			First entity involved in this MetaRelation (all the Relations in it are between the same entities)
		domain2 : NX.Entity
			Second entity involved in this list of relations (MetaRelation).
		name: str
			Name of the corresponding MetaRelation
		ignore_index: int
			Value corresponding to missing values. Used to allow fast runs on GPUs and minibatching even with different percentages of missing values among the Relations/SubDatasets in the same MetaRelation/MetaDataset.

		Returns
		-------
		
		"""
		self.name = name
		self.ignore_index = ignore_index
		self.side1 = side1
		self.side2 = side2
		self.datasetList = datasetList
		self.domain1 = domain1
		self.domain2 = domain2

	def getTypes(self):
		r = []
		for i in self.datasetList:
			r.append(i.type)
		return r

	def countBalance(self):
		r = []
		for i in self.datasetList:
			r.append(i.balance)
		return r

	def countInstances(self):
		r = []
		for i in self.datasetList:
			r.append(i.estSize)
		return r

	def getEstSize(self):
		return sum(self.countInstances())

	def getEstBatchSizeForXsamples(self, targetDomain1, samplesPerBatch):
		assert samplesPerBatch > 0
		perc = samplesPerBatch / float(len(targetDomain1))
		# print(len(targetDomain1))
		# print(len(self.domain1))
		res = int(max(1, perc * float(len(self.domain1))))
		print (" Foreseen batch size: ", res)
		return res

	def getEstBatchSizeForXsamples2(self, numSamples): #future daniele, pay attention to this, may be working bady
		assert numSamples > 10
		tot = self.getEstSize()/len(self.domain2)
		res = max(1,tot / numSamples)
		print( " Foreseen batch size: ", res)
		return res

	def __getitem__(self, idx):
		tmp = []
		for d in self.datasetList:
			tmp.append(d[idx])
		tmp = self.mergeDataSimple(tmp, idx)
		#print tmp
		#raw_input()
		return tmp
	
	def mergeDataSimple(self, v, idx):
		x1 = []
		x2 = []
		y = []
		xside1 = []
		xside2 = []
		#da sistemare qui sotto
		for count, ds in enumerate(v):
			#print count
			assert len(ds[0]) == len(ds[1]) and len(ds) == 2
			tmp1 = np.array([idx]*len(ds[0]), dtype=np.int32) #BEWARE! you might have more instances TODO
			#print "x:", tmp1.shape, len(ds[0])
			x1.append(tmp1)
			x2.append(ds[0])
			tmp = np.ones((len(ds[0]), len(v)))* self.ignore_index
			tmp[:,count] = ds[1]

			if self.side1 != None and len(ds[0]) > 0:
				#print self.side1[idx]
				#print len(ds[0])
				for s in range(0, len(ds[0])):
					xside1.append( self.side1[idx])
				
			if self.side2 != None and len(ds[0]) > 0:
				tmp1 = []
				for s in ds[0]:
					xside2.append(self.side2[s])
			y.append(tmp)
			#print tmp.shape
		if self.side1 != None and len(xside1) > 0:
			xside1 = np.vstack(xside1)
		else:
			xside1 = []
		if self.side2 != None and len(xside2) > 0:
			xside2 = np.vstack(xside2)
		else:
			xside2 = []
		#print xside2
		if self.side1 != None or self.side2 != None:
			#print len(xside1), len(xside2)
			#print xside1.shape, xside2.shape
			return np.hstack(x1), np.hstack(x2), np.vstack(y), xside1, xside2
		else:		
			return np.hstack(x1), np.hstack(x2), np.vstack(y)

	def __len__(self):
		return len(self.domain1)	


