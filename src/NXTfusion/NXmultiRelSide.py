#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ppiPred.py
#  
#  Copyright 2017 Daniele Raimondi <eddiewrc@alice.it>
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
import os, sys, gc, marshal, copy, random, time, socket, math
import scipy.stats as st
import NXTfusion.NXFeaturesConstruction as FC
import NXTfusion.NXMetaLoaderUtils as MLU
import NXTfusion.NXCollateUtils as Collate
from sys import stdout
import numpy as np
import torch as t
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import NXTfusion.NXLosses as L
import NXTfusion.NXDatasetUtils as D
import NXTfusion.NXTfusion as NX
from NXTfusion.Logger import MetaLogger
from tqdm import tqdm
#np.seterr(all="raise")

class NNwrapper():
	"""
	Class that wraps a t.nn.Module (pytorch module) and uses scikit-learn-like methods such as .fit() and .predict() to train and test it.
	"""

	def __init__(self, model, dev, ignore_index, initialEpoch=0, nworkers = 0):
		"""
		Constructor for the NNWrapper class, which facilitates and standardizes the training of pytorch neural networks.

		Parameters
		----------
		model : t.nn.Module 
			The pytorch Neural Network that should be trained or tested.
		def : t.device
			The device on which the model should run. E.g. t.device("cuda") or t.device("cpu:0")
		ignore_index : int
			The ignore index value that will be used to mark "missing values" and "N/A" on partially observed matrices, in order to let the corresponding loss ignore those instances.
		

		Returns
		-------
		"""
		self.model = model
		self.ignore_index = ignore_index
		self.initialEpoch = initialEpoch +1
		self.logger = None
		self.save_model_every = 5
		self.nworkers = nworkers
		self.model.to(dev)
		self.dev = dev
		print( "***** Model device: ", self.dev)

	def processDatasets(self, DS:list):

		""" 
		This method takes the external Entity Relation graph representation, in the form of one MetaRelation at a time and converts it
		into lower level data structures to be used within the wrapper, 
		creating a MetaDataset structure from the ER representation passed as input. This structure mimics the ERgraph, but it's suitable for efficient multi-task mini batching during training.
		This function is used internally by the NNwrapper and does not need to be called by the user.
		

		Parameters
		----------
		DS : MetaRelation 
			
		Returns
		-------
		DS : MetaRelation
			The original MetaRelation without the data matrices, in an attempt to save space. (still have to run benchmarks on it)
		datasets : list of SubDataset
		losses : list of losses
		refSize : size of the target matrix (to be removed)
		:meta private:
		"""

		datasets = []
		losses = []
		refSize = None
		print ("Loading datasets...")
		for i, ds in enumerate(DS):
			print ("Working on ds %s..." % ds["name"])
			#if ds["type"] == "DS": #realDataset
			datasets.append(D.SubDataset(ds["data"].data))
			DS[i].__delitem__("data")		
			if ds["name"] == "target":
				refSize = datasets[-1].estSize
			losses.append(ds["loss"])
		print ("Done.")
		gc.collect()
		#assert refSize != None
		return DS, datasets, losses, refSize

	def saveModel(self, e:int):
		"""
		Method that stores the trained model at a certain iteration. Used internally.

		Parameters
		----------
		e : int
			Epoch number. The model is automatically saved using the model name and the epoch number using t.save function.

		"""
		if e == True or e % self.save_model_every == 0:
			t.save(self.model.gnn.state_dict(), self.model.name+".iter_"+str(e)+".t")
			print ("Store model ", e)			
			stdout.flush()

	def countParams(self, parameters:list) -> int:
		"""
		Method that counts the number of trainable parameters in the model.

		Parameters
		----------
		parameters : iterable
			The iterable containtaining the pytorch model parameters.

		Returns
		-------
		Number of parameters (int)
		
		"""

		p = []
		for i in parameters:
			p+= list(i.data.cpu().numpy().flat)
		print ('Number of parameters=',len(p))
		return len(p)

	def getRelationWeights(self, relData, datasets, refSize):
		"""

		:meta private:
		"""
		relationWeight = []
		for i, ds in enumerate(relData):
			if type(ds["relationWeight"]) != str:
				relationWeight.append(ds["relationWeight"])
			elif ds["relationWeight"] == "relativeToTarget":
				w = refSize/float(datasets[i].estSize)
				relationWeight.append(w)
			print ("Rel %d, name: %s, type: %s, loss: %f" % (i, ds["name"], str(ds["relationWeight"]), relationWeight[-1]))
		tmp = t.tensor(relationWeight, dtype=t.float, device=self.dev)
		return tmp

	def computeLosses(self, y, yp, losses, relationData, weightRelations):
		"""
		This function computes the losses for the entire ER graph, by iterating through them.  Used internally.

		Parameters
		----------
		y : t.tensor
			Pytorch tensor containing the labels
		yp : t.tensor
			Pytorch tensor containing the predictions
		losses : list 
			list of losses (LossWrapper or t.nn.Module)
		relationData : list
			list of MetaRelations
		weightRelations : list
			list of weights associated to each loss

		Returns
		-------
		loss : real
			total loss
		tmpLoss : list
			list containing the losses associated to each Relation
		:meta private:
		"""
		tmpLoss = []
		assert len(losses) == len(relationData) == len(weightRelations)
		for ci, l in enumerate(losses):
			if l.type == "binary":
				assert l.type == relationData[ci]["task"]
				tmpLabel = y[:,ci:ci+1,].squeeze()
			elif l.type == "regression":
				assert l.type == relationData[ci]["task"]
				tmpLabel = y[:,ci:ci+1,].squeeze()
			else:
				raise Exception("unrecognized")
			tmpLoss.append(weightRelations[ci] * losses[ci](yp[:,ci:ci+1], tmpLabel))
		#print tmpLoss
		return sum(tmpLoss), tmpLoss 	

	def printBatchesLog(self, rel, e, bi, errTotOld, errTot, totLen, epochTime, loadTime, forwTime, LossTime, start, batch_size, mute = True):

		""" This class simplifies the live logging of the batches. If muted, it will only signal excessively long loading times.

		Parameters
		----------
		TODO

		Returns
		-------
		:meta private:
		"""
		if mute:
			if loadTime > 0.5:
				print( "WARNING: loadTime %3.2fs !" % loadTime)
		else:
			sys.stdout.write("rel: %d, e: %d, b: %d/%d, %3.2f%% (T:%.3fs, e:%.3fs, l:%.3fs, f:%.3fs, b:%.3fs)\n" % (rel, e, bi, totLen, 100*(bi/float(totLen)), time.time() - start, epochTime, loadTime, forwTime, LossTime, )	)
			sys.stdout.flush()	# T: total, e: epochtime, l: load, f: forward, b: back

	def printLossLog(self, e, datasetsList, errors, start, end, errTotOld, errTot):
		"""
		This function prints the logs at each epoch. Used to make the code less cumbersome to read.

		Parameters
		----------
		TODO

		Returns
		-------
		
		:meta private:
		"""
		sys.stdout.write("\n epoch %d,"% e)
		percRed = ((errTotOld-errTot)/float(errTotOld))*100.0	
		#print errors, len(errors), len(tmpLoss)
		sys.stdout.write(" ERRORTOT: %f (%fs) %2.3f%%\n" % (errTot, end-start, percRed))
		for i, rel in enumerate(errors):	
			sys.stdout.write("Relation: %d\n" % i)
			for li, l in enumerate(rel):
				sys.stdout.write("\t Mat: %s: error: %.2f\n" % (datasetsList[i][li]["name"],
										float(sum(rel[li]) / len(rel[li]))))
		sys.stdout.write("\n")
		sys.stdout.flush()
	
	def buildTensorboardLog(self, errTot, lossScores, relationList, te):
		"""
		:meta private:
		"""
		if self.logger == None:
			print (" WARNING: Logger not initializated, skipping")
			return
		info = {"errTot":errTot}
		for r in relationList:
			for i, l in enumerate(lossScores):
				info[r[i]["name"]] = l
		info["timePerEpoch"] = te

		for tag, value in info.items():
			logger.scalar_summary(tag, value, step+1)

		# (2) Log values and gradients of the parameters (histogram)
		for tag, value in self.model.named_parameters():
			#print
			#print tag, value, value.grad					
			tag = tag.replace('.', '/')
			logger.histo_summary(tag, to_np(value), step+1)
			logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)

		return info
		
	def fit(self, relationList, epochs = 100, batch_size=500,
		save_model_every=10, LOG=False, MUTE = True,
		weight_decay=0.001):
		"""
		Function that performs the training of the wrapped pytorch model. 
		It is analogous to scikit-learn .fit() method.


		Parameters
		----------
		relationList : ERgraph
		epochs : int
			Number of epochs
		batch_size : int
			batch size during training
		save_model_every : int
			Stores the model every int epochs

		Returns
		-------
		

		"""
		print ("self.dev: ", self.dev)
		self.model.train()
		print ("Training mode: ", self.model.training)
		FAST = False 
		self.save_model_every = 20
		if LOG:
			self.logger = MetaLogger(self.model, port = 6001)
		########DATASET###########
		print ("Processing %d relations " % len(relationList))
		datasetsList = []
		lossesList = []
		weightRelations = []
		refSize = None
		for i, relData in enumerate(relationList):
			r, datasets, losses, tmpRefSize = self.processDatasets(relData) #relations, pytorchDatasets, losses
			if tmpRefSize != None:
				assert refSize == None
				refSize = tmpRefSize
			datasetsList.append(D.MetaDataset(datasets, r[0]["domain1"], r[0]["domain2"], relData.name, self.ignore_index, relData.side1, relData.side2))
			weightRelations.append(self.getRelationWeights(relData, datasets, refSize))
			relationList[i] = r
			lossesList.append(losses)
			gc.collect()
		#######MODEL##############


		batches = {}
		for i in range(len(relationList)):
			batches[i] = [{}, {}]
		parameters = self.model.getParameters()
		self.countParams(parameters)	
		self.model.train()
		print( "Training mode: ", self.model.training)

		print ("Start training")
		########OPTIMIZER##########	
		self.learning_rate = 1e-3
		if  next(self.model.parameters()).is_cuda: 
			assert next(self.model.parameters()).is_cuda and "cuda" in self.dev
		optimizer = t.optim.Adam(parameters, lr=self.learning_rate, weight_decay=weight_decay)
		scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
		
		########DATALOADER#########
		print( "Creating dataloaders for %s..." % datasetsList)
		loaders = []
		for ds in datasetsList:
			# print(relationList[0][0]["domain1"])

			loaders.append(DataLoader(ds,
				batch_size=ds.getEstBatchSizeForXsamples(relationList[0][0]["domain1"], batch_size), shuffle=True, sampler=None, num_workers=self.nworkers, collate_fn=Collate.metaCollate, pin_memory=False))
		metaLoader = MLU.MetaLoader(loaders)	
		e = self.initialEpoch
		assert len(datasetsList) == len(loaders) == len(metaLoader.loaders)
		errTot = 0
		###############TRAINING ITERATIONS#############
		for e in tqdm(range(self.initialEpoch, self.initialEpoch + epochs)): #while e < epochs + self.initialEpoch:

			errTotOld = errTot
			errTot = 0
			errors = []
			for r in relationList:
				errors.append([[]]*len(r))
			c = 0	
			start = time.time()		
			bi = []
			for b in datasetsList:
				bi.append(0)
			tLoad1 = time.time()	# start epoch time
			skipped = False
			for sample in metaLoader:
				tLoad2 = time.time() # end load time, start forward time
				lossTot = 0
				optimizer.zero_grad()	
 				#D.printSampleSizes(sample, relationList)
				for r, rel in enumerate(sample):
					#if self.CUDA:
					#	t.cuda.empty_cache()
					if len(rel[0]) == 0:
						print("Skipping")
						skipped=True
						continue
					bi[r] += loaders[r].batch_size
					if len(rel) > 3: #if side information is present
						x1, x2, y, side1, side2 = rel
						#	print side1, side2
						x1 = t.tensor(x1, dtype=t.long, device=self.dev)
						x2 = t.tensor(x2, dtype=t.long, device=self.dev)
						y = t.tensor(y, dtype=t.float, device=self.dev)
						if type(side1) != type(None):
							#print "side1 not none"
							#print side1
							side1 = t.tensor(side1, dtype=t.float, device=self.dev)
						if type(side2) != type(None):
							#print side2.shape
							side2 = t.tensor(side2, dtype=t.float, device=self.dev).squeeze()
							#print "side2 not none"
					else:
						x1, x2, y = rel
						# x1_k = x1.tostring()#np.array2string(x1, max_line_width=10000)
						# x2_k = x2.tostring()#np.array2string(x2, max_line_width=10000)
						# if x1_k in batches[r][0]:
						# 	x1 = batches[r][0][x1_k]
						# else:
						x1 = t.tensor(x1, dtype=t.long, device=self.dev)
							# batches[r][0][x1_k] = x1
						# if x2_k in batches[r][1]:
						# 	x2 = batches[r][1][x2_k]
						# else:
						x2 = t.tensor(x2, dtype=t.long, device=self.dev)
							# batches[r][1][x2_k] = x2
						# x2 = t.tensor(x2, dtype=t.long, device=self.dev)
						y = t.tensor(y, dtype=t.float, device=self.dev)
						#print y.tolist(), r
						#raw_input()
					#print "Rel: %d, size: %s" % (r, str(x1.size()))
					#print "Rel %d, batch size %s" % (r, x1.size())	
					tForw1 = time.time()
					assert datasetsList[r].name == relationList[r].name
					if len(rel) > 3:
						yp = self.model.forward(relationList[r].name, x1, x2, NX.SafeVariable(side1), NX.SafeVariable(side2))
					else:
						yp = self.model.forward(relationList[r].name, x1, x2)
					if len(yp.size()) == 1:
						yp = yp.unsqueeze(0)
					tForw2 = time.time() #end forward time, start loss time
					tLoss1 = tForw2
					tmpL, tmpLoss = self.computeLosses(y, yp, lossesList[r], relationList[r], weightRelations[r])
					errTot += tmpL.data.cpu()
					lossTot += tmpL
					for li, l in enumerate(tmpLoss):					
						errors[r][li] += [l.data.cpu()]
					tLoss2 = time.time() #end epoch time
					# self.printBatchesLog(r, e, bi[r], errTotOld, errTot, len(datasetsList[r]), time.time()-tLoad1, tLoad2-tLoad1, tForw2-tForw1, tLoss2-tLoss1, start, batch_size, mute = MUTE)

				if skipped:
					continue
				lossTot.backward()	
				optimizer.step()
				if FAST and c > 2:
					break
				c+=1
				tLoad1 = time.time()
				#if self.CUDA: #makes things slower
				#	t.cuda.empty_cache()
			end = time.time()	
			self.printLossLog(e, relationList, errors, start, end, errTotOld, errTot)
			if LOG:
				self.logger.writeTensorboardLog(e, errTot, errors, relationList, end-start)
			scheduler.step(float(errTot))
			# self.saveModel(e)
			e += 1	
		#self.saveModel(True)
		if LOG:
			self.logger.shutdown()
	
	def predict(self, ERgraph, X, metaRelationName, relationName, sidex1=None,
		    sidex2=None, batch_size=500, plotGraph=False, get_latent_embeddings=False):
		"""
		Function that performs the training of the wrapped pytorch model. It is analogous to scikit-learn .predict() method.



		Parameters
		----------
		ERgraph : ERgraph
		X : list
			List containing the 2D coordinates of the positions that should be predicted in the ERgraph.metaRelationName.relationName Relation.
		metaRelationName : str
			Name of the MetaRelation that contains the target relation
		relationName : str
			Name of the relation that you want to predict
		batch_size : int
			batch size during prediction

		Returns
		-------
		yp : list
			List containing the predictions for the target Relation

		"""
		targetMetaRel = ERgraph[metaRelationName]["metaRelation"]
		assert targetMetaRel != None, "ERROR: "+ metaRelationName+" is not a MetaRelation in the ER graph!"
		targetRelPos = targetMetaRel.getPos(relationName)
		assert targetRelPos != None, "ERROR: "+relationName+"is not a Relation in "+metaRelationName+" in the ERgraph!"
		print ("Rel in pos", targetRelPos)
		print ("self.DEVICE: ", self.dev)
		self.model.eval()
		print( "Training mode: ", self.model.training)
		if plotGraph:
			from pytorchUtils.torchgraphviz1 import make_dot, make_dot_from_trace
		
		print ("Predicting...")
		if sidex1 == None and sidex2 == None:
			dataset = D.PredictionDataset(X)
		else:
			dataset = D.PredictionDatasetSide(X, sidex1, sidex2)
		#predDataset = D.PredictMetaDataset([dataset], predRel.domain1, predRel.domain2, predRel.side1, predRel.side2)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=0, collate_fn=Collate.predictMetaCollate)
		yp = []			
		first = True
		if not get_latent_embeddings:
			for sample in tqdm(loader):
				if len(sample) > 3:
					x1, x2, sx1, sx2 = sample
					x1 = t.tensor(x1, dtype=t.long, device=self.dev)
					x2 = t.tensor(x2, dtype=t.long, device=self.dev)
					if type(sx1) != type(None):
						sx1 = t.tensor(sx1, dtype=t.float, device=self.dev)
					if type(sx2) != type(None):
						sx2 = t.tensor(sx2, dtype=t.float, device=self.dev)
				else:
					x1, x2 = sample
					x1 = t.tensor(x1, dtype=t.long, device=self.dev)
					x2 = t.tensor(x2, dtype=t.long, device=self.dev)
				if len(sample) > 3:
					pred = self.model.forward(metaRelationName, x1, x2, NX.SafeVariable(sx1), NX.SafeVariable(sx2))
				else:
					pred = self.model.forward(metaRelationName, Variable(x1), Variable(x2))
				if first and plotGraph:
					first = False
					make_dot(pred.mean(), params=dict(self.model.named_parameters()))
				if len(pred.data.size()) == 1:
					pred.data = pred.data.unsqueeze(1)
					yp += [pred.data.item()]
				else:
					yp += pred.data[:,targetRelPos].cpu().tolist()

		else:

			pred = self.model.get_latent_repr(metaRelationName)
			yp = pred

		return yp
