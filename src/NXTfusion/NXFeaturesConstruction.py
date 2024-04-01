#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  PPIpredUtils.py
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
import random, time, math, torch
import torch as t
import numpy as np
import NXTfusion.NXTfusion as NX
import NXTfusion.DataMatrix as DM
from multipledispatch import dispatch
from scipy.sparse import coo_matrix

def getOutputs(yp, c):
	"""
	:meta private:
	"""
	return yp[:,c:c+1]

def buildPytorchSide(data, domain, expectedLen = 20, sideDtype=np.float32):
	""" This function builds the data structure containing the side information 
	
		The data structure is a {} indicized with the domain numeric names.

		sideX = {domainName_numeric : numpy32_feats}
	:meta private:
	"""
	x = {}
	i = 0
	while i < len(domain):
		x[i] = []
		i+=1

	
	missing = 0
	for i in domain:
		#print i
		try:
			x[domain[i]].append(np.array(data[i], dtype=sideDtype))	
		except:
			x[domain[i]].append(np.zeros(expectedLen, dtype=sideDtype))	
			missing +=1
		#	print i, x[lookup[i]]
		#print len(x[lookup[i]][0]) #== 50
		#raw_input()
	if missing > 0:
		print( "WARNING: %d missing proteins in side info." % missing)
	return x


@dispatch(np.ndarray, NX.Entity, NX.Entity)
def buildPytorchFeats(data:np.ndarray, domain1:NX.Entity, domain2:NX.Entity, side1 = None, side2 = None) -> (list, list, list, list, list):
	"""
	This function is used to produce the input for the NNwrapper.predict() method, at prediction time. It produces the inputs necessary to predict the output for certain cells (or the entire matrix) for a given relation.

	This version of the method takes as prediction target a (dense) numpy matrix.

	Parameters
	----------
	data : np.ndarray
		Numpy matrix representing the target. This form of the method is more useful when the entire matrix needs to be predicted. The actual values in the matrix are provided as "labels" in output, but are ignored during prediction.
	domain1 : NX.Entity
		Entity representing the objects on the dimension 0 of the data matrix
	domain2 : NX.Entity
		Entity representing the objects on the dimension 1 of the data matrix
	
	Returns
	-------
	x : list of tuples [(i,j),(k,j),...]
		List containing the pairs of of objects belonging to domain1 and domain2 that needs to be predicted.
	y : list
		Values of the input data matrix corresponding to the pairs of object in x
	corresp: list of tuples
		List of tuples containing the corresponding names of the pairs of objects listed in x.
	"""
	assert data.shape[0] == len(domain1) and data.shape[1] == len(domain2)
	if side1 != None:
		assert len(side1) == data.shape[0]
	if side2 != None:
		assert len(side2) == data.shape[1]
	x = []
	y = []
	sx1 = []
	sx2 = []
	corresp = []
	
	i = 0
	while i < data.shape[0]:
		j = 0
		while j < data.shape[1]:
			#print i #(('Q96IY1', 'Q9UGM6'), 0)
			#raw_input()
			corresp.append((domain1[i], domain2[j]))
			x.append([i,j])
			if side1 != None:
				sx1.append(side1[i])
			if side2 != None:
				sx2.append(side2[j])
			y.append(data[i,j])
			j+=1
		i+=1
	assert len(x) == len(y)	
	if side1 == None:
		sx1 = None
	if side2 == None:
		sx2 == None
	if side1 == None and side2 == None:
		return x, y, corresp
	else:
		return x, y, sx1, sx2, corresp

@dispatch(dict, NX.Entity, NX.Entity)
def buildPytorchFeats(data:dict, domain1:NX.Entity, domain2:NX.Entity, side1 = None, side2 = None):
	"""
	This function is used to produce the input for the NNwrapper.predict() method, at prediction time. It produces the inputs necessary to predict the output for certain cells (or the entire matrix) for a given relation.

	This version of the method takes as prediction target a dict containing the pairs of ojects that need to be predicted. This is useful when only relatively few cells of the matrix need to be predicted (sparse prediction).

	Parameters
	----------
	data : dict {(obj[i],obj[j]):value1, (obj[i],obj[k]):value2, ...}
		Dict representing the target cells of the matrix that need to be predicted. The actual values in the matrix are provided as value associated to each pair of objects in the dict, but are ignored during prediction.
	domain1 : NX.Entity
		Entity representing the objects on the dimension 0 of the data matrix
	domain2 : NX.Entity
		Entity representing the objects on the dimension 1 of the data matrix
	
	Returns
	-------
	x : list of tuples [(i,j),(k,j),...]
		List containing the pairs of of objects belonging to domain1 and domain2 that needs to be predicted.
	y : list
		Values of the input data matrix corresponding to the pairs of object in x
	corresp: list of tuples
		List of tuples containing the corresponding names of the pairs of objects listed in x.
	"""
	x = []
	y = []
	sx1 = []
	sx2 = []
	corresp = []
	
	for i in data.items():
		#print i #(('Q96IY1', 'Q9UGM6'), 0)
		#raw_input()
		corresp.append(i[0])
		x.append([domain1[i[0][0]], domain2[i[0][1]]])
		if side1 != None:
			sx1.append(side1[domain1[i[0][0]]])
		if side2 != None:
			sx2.append(side2[domain2[i[0][1]]])
		y.append(i[1])
	assert len(x) == len(y)	
	if side1 == None:
		sx1 = None
	if side2 == None:
		sx2 == None
	if side1 == None and side2 == None:
		return x, y, corresp
	else:
		return x, y, sx1, sx2, corresp

@dispatch(DM.DataMatrix)
def buildPytorchFeats(datam:DM.DataMatrix, side1 = None, side2 = None):
	"""
	This function is used to produce the input for the NNwrapper.predict() method, at prediction time. It produces the inputs necessary to predict the output for certain cells (or the entire matrix) for a given relation.

	This version of the method takes as prediction target a DataMatrix object.

	Parameters
	----------
	datam : DataMatrix
		DataMatrix object containing the matrix representing the predidction target. The actual values in the observed cells in the DataMatrix are provided as "labels" y in output, but are ignored during prediction.
		
	Returns
	-------
	x : list of tuples [(i,j),(k,j),...]
		List containing the pairs of of objects belonging to domain1 and domain2 that needs to be predicted.
	y : list
		Values of the input data matrix corresponding to the pairs of object in x
	corresp: list of tuples
		List of tuples containing the corresponding names of the pairs of objects listed in x.
	"""
	data = datam.toHashTable()
	domain1 = datam.ent1
	domain2 = datam.ent2
	x = []
	y = []
	sx1 = []
	sx2 = []
	corresp = []
	
	for i in data.items():
		#print i #(('Q96IY1', 'Q9UGM6'), 0)
		#raw_input()
		corresp.append(i[0])
		x.append([domain1[i[0][0]], domain2[i[0][1]]])
		if side1 != None:
			sx1.append(side1[domain1[i[0][0]]])
		if side2 != None:
			sx2.append(side2[domain2[i[0][1]]])
		y.append(i[1])
	assert len(x) == len(y)	
	if side1 == None:
		sx1 = None
	if side2 == None:
		sx2 == None
	if side1 == None and side2 == None:
		return x, y, corresp
	else:
		return x, y, sx1, sx2, corresp

@dispatch(coo_matrix, NX.Entity, NX.Entity)
def buildPytorchFeats(data:coo_matrix, domain1:NX.Entity, domain2:NX.Entity, side1 = None, side2 = None):
	"""
	This function is used to produce the input for the NNwrapper.predict() method, at prediction time. It produces the inputs necessary to predict the output for certain cells (or the entire matrix) for a given relation.

	This version of the method takes as prediction target a dict containing the pairs of ojects that need to be predicted. This is useful when only relatively few cells of the matrix need to be predicted (sparse prediction).

	Parameters
	----------
	data : coo_matrix 
		Sparse matrix representing the target cells of the matrix that need to be predicted. The actual values in the matrix are provided as value associated to each pair of objects in the dict, but are ignored during prediction.
	domain1 : NX.Entity
		Entity representing the objects on the dimension 0 of the data matrix
	domain2 : NX.Entity
		Entity representing the objects on the dimension 1 of the data matrix
	
	Returns
	-------
	x : list of tuples [(i,j),(k,j),...]
		List containing the pairs of of objects belonging to domain1 and domain2 that needs to be predicted.
	y : list
		Values of the input data matrix corresponding to the pairs of object in x
	corresp: list of tuples
		List of tuples containing the corresponding names of the pairs of objects listed in x.
	"""
	x = []
	y = []
	sx1 = []
	sx2 = []
	corresp = []
	d = {}
	data = data.todok()
	for i in data.items():
		d[i[0]] = i[1]

	for i in data.items():
		#print i #(('Q96IY1', 'Q9UGM6'), 0)
		#raw_input()
		corresp.append(i[0])
		x.append([domain1[i[0][0]], domain2[i[0][1]]])
		if side1 != None:
			sx1.append(side1[domain1[i[0][0]]])
		if side2 != None:
			sx2.append(side2[domain2[i[0][1]]])
		y.append(i[1])
	assert len(x) == len(y)	
	if side1 == None:
		sx1 = None
	if side2 == None:
		sx2 == None
	if side1 == None and side2 == None:
		return x, y, corresp
	else:
		return x, y, sx1, sx2, corresp

def main():
	"""
	:meta private:
	"""	
	return 0

if __name__ == '__main__':
	main()

