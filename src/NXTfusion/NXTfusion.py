#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pytorchLossUtils.py
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
import numpy as np
import torch as t
from torch.autograd import Variable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from multipledispatch import dispatch

def SafeVariable(v):
	""" 
		Utility function that helps checking types.

		Parameters
		----------
		v : object
			Input variable

		Returns
		-------
		Returns None if v is none, raises exception if v is 0 and returns v otherwise.
	:meta private:
	"""

	if type(v) == type(None):
		return None
	elif len(v) == 0:
		raise Exception("Very strange to have empty vars here")
	else:
		return v

def f():
	"""
	:meta private:
	"""
	pass

class Entity(object):

	"""
	Class representing the Entity concept. 
	"""
	#@dispatch(str, list, type)
	def __init__(self, name, domain, dtype=np.int32):
		""" 
			Constructor for the Entity class.

			Parameters
			----------
			name : str
				Name of the Entity (use a mnemonic name describing the class of objects represented by the Entity)
			domain : iterable (list) containing str
				List of the possible objects belonging to this class (e.g. patients IDs, proteins Uniprot identifiers, ...). It is an unique identifier naming (with a string) all the objects composing the domain of the Entity.
			dtype : np.dtype
				Smallest possible numpy type able to uniquely enumerate all the objects. len(domain) < max_number_representable(dtype).
			
			Returns
			-------
		"""
		assert isinstance(domain, Iterable)
		assert len(domain) > 0
		self.dtype = dtype
		self.name = name
		self.domain = domain
		self.lookup = {}
		i = 0
		while i < len(domain):
			self.lookup[domain[i]] = i
			i+=1
		assert len(self.lookup) == len(self.domain)

	def __next__(self):
		return self.domain.next()

	next = __next__

	def __iter__(self):
		return iter(self.domain)

	@dispatch(str)
	def __getitem__(self, x):
		""" Method that	returns the numeric value internally associated to each object in the Entity class.

			Parameters
			----------
			x : str
				String name of a specific object in the Entity.

			Returns
			primary key : int
			-------
		"""
		return self.lookup[x]
	
	@dispatch(int)
	def __getitem__(self, x):
		""" Method that	returns the str name of the object with primary key x.

			Parameters
			----------
			x : int
				Primary key (unique id) of an object in the domain represented by the Entity.

			Returns
			name of the object : str
			-------
		"""
		#print(x, len(self.domain))
		assert x < len(self.domain)
		return x

	@dispatch(np.int16)
	def __getitem__(self, x):
		""" Method that	returns the str name of the object with primary key x.

			Parameters
			----------
			x : int
				Primary key (unique id) of an object in the domain represented by the Entity.

			Returns
			name of the object : str
			-------
		"""
		#print(x, len(self.domain))
		assert x < len(self.domain)
		return x

	@dispatch(np.int64)
	def __getitem__(self, x):
		""" Method that	returns the str name of the object with primary key x.

			Parameters
			----------
			x : int
				Primary key (unique id) of an object in the domain represented by the Entity.

			Returns
			name of the object : str
			-------
		"""
		#print(x, len(self.domain))
		assert x < len(self.domain)
		return x

	@dispatch(np.int32)
	def __getitem__(self, x):
		""" Method that	returns the str name of the object with primary key x.

			Parameters
			----------
			x : int
				Primary key (unique id) of an object in the domain represented by the Entity.

			Returns
			name of the object : str
			-------
		"""
		#print(x, len(self.domain))
		assert x < len(self.domain)
		return x
		#return self.domain[x]
	
	def has_key(self, x):
		return x in self.lookup

	def __len__(self):
		return len(self.domain)

class Relation(dict):
	"""
	Class that represent a relation (matrix in MF terms) with all its parameters and functions.

	
	"""
	#@dispatch(str, Entity, Entity, DM.DataMatrix, str, )
	def __init__(self, name:str, domain1:Entity, domain2:Entity, data, task:str, loss:t.nn.Module, relationWeight:float, side1=None, side2 = None, path=None):
		""" Constructor for the Relation class..

			Parameters
			----------
			name : str
				Mnemonic name of the specific relation/matrix.
			domain1 : Entity
				Entity1 involved in the relation (on dimension 0)
			domain2 : Entity
				Entity2 involved in the relation (on dimension 1)
			data : DataMatrix
				DataMatrix object containing the matrix describing this relation
			task : str ["regression", "binary"]
				Type of prediction task associated to this relation. "Regression" for real valued predictions, "binary" for binary classification.
			loss : NX.NXLosses or t.nn.Module
				Pytorch-like loss module corresponding to the loss that must be used to compute the reconstruction error for this relation.
			relationWeight : float
				A relation-specific weight that will multiply the loss score during training.

			Returns
			-------
		"""
		super(Relation, self).__init__({})	
		assert type(name) == str
		self["name"] = name
		#assert type1 == "DS" or type1 == "OD"
		#d["type"] = type1
		#assert type(domain1) == Entity
		self["domain1"] = domain1
		assert type(domain2) == Entity
		self["domain2"] = domain2
		self["data"] = data
		assert task == "regression" or task == "binary"
		self["task"] = task
		assert relationWeight == "relativeToTarget" or type(relationWeight) == int or type(relationWeight) == float
		self["relationWeight"] = relationWeight
		self["loss"] = loss

class MetaRelation(object): #it's basically a list with constraints on what you can add

	"""Constructor for the Relation class. The Meta Relation represents multi-relations between the same entities (used for example in tensor factorization).

	As a convention, we recommend to use names in the form ENT1-ENT2.

	The domains must be the same for each relation in it, since the MetaRelation defines a tensor where the dimension 0 and 1 represent the same entities for all the matrices involved in the tensor.
	Can allow side info (common to all relations in it).

	"""

	def __init__(self, name,  domain1, domain2, side1=None, side2=None, relations=[], prediction=False):
		"""Constructor for the MetaRelation class.

		Parameters
		----------
		name : str
			Mnemonic name of the specific relation/matrix.
		domain1 : Entity
			Entity1 involved in the relation (on dimension 0)
		domain2 : Entity
			Entity2 involved in the relation (on dimension 1)
		relations : list of NX.Relation objects
			List of the relations involved in this MetaRelation (tensor)
		
		Returns
		-------
		"""
		self.name = name
		if side2 != None:
			assert len(side2) == len(domain2)
		self.side2 = side2
		if side1 != None:
			assert len(side1) == len(domain1)
		self.side1 = side1
		assert type(domain1) == Entity
		self.domain1 = domain1
		assert type(domain2) == Entity
		self.domain2 = domain2

		self.relationList = []
		for r in relations:
			self.append(r)

	def append(self, r):
		"""
		Method that adds a Relation object to an existing MetaRelation
		Parameters
		----------
		r : Relation
				
		Returns
		-------
		"""

		#print type(r)
		assert isinstance(r, Relation)
		assert self.domain1 == r["domain1"]
		assert self.domain2 == r["domain2"]
		self.relationList.append(r)

	#@dispatch(str)
	def getPos(self, x):
		for c, i in enumerate(self.relationList):
			if i["name"] == x:
				return c
		print ("MetaRelation %s does not contain relation named %s"% (self.name, x))
		return None


	@dispatch(str)
	def __getitem__(self, x:str)->Relation:
		"""
		Getitem method that searches by Relation.name

		Parameters
		----------
		x : str
			The name of a Relation in this MetaRelation

		Returns
		-------
		The target Relation or None
		"""

		for i in self.relationList:
			if i["name"] == x:
				return i
		print ("MetaRelation %s does not contain relation named %s"% (self.name, x))
		return None

	@dispatch(int)
	def __getitem__(self, x:int) -> Relation:
		"""
		Getitem method that searches by position of the target Relation in the tensor/MetaRelation.

		Parameters
		----------
		x : int
			The position (index) of a Relation in this MetaRelation

		Returns
		-------
		The target Relation or None
		"""

		return self.relationList[x]

	def __next__(self):
		return self.relationList.next()

	next = __next__

	def __iter__(self):
		return iter(self.relationList)

	def pop(self, pos):
		return self.relationList.pop(pos)

	def __len__(self):
		return len(self.relationList)

class ERgraph(list):
	"""
	Class that represents the entire Entity-Relation graphs, namely a list of MetaRelations. Each MetaRelation might contain multiple Relations.
	"""
	def __init__(self, entityList:list, name = ""):
		"""
		Constructor for the ERgraph (Entity-Relation Graph) object.

		Parameters
		----------
		entityList : list of MetaRelations
			List of MetaRelations that describe the full Entity Relation graph.

		Returns
		-------
		"""

		self.graph = []
		self.name = name
		self.lookup = {}
		for i, rel in enumerate(entityList):
			assert type(rel) == MetaRelation
			self.graph.append({"name":rel.name, "pos": i, "lenDomain1":len(rel.domain1), "lenDomain2":len(rel.domain2), "arity":len(rel), "metaRelation":rel}) #TODO:add side
			print({"name":rel.name, "pos": i, "lenDomain1":len(rel.domain1), "lenDomain2":len(rel.domain2), "arity":len(rel), "metaRelation":rel})
			self.lookup[rel.name] = self.graph[i]
		super(ERgraph, self).__init__(entityList)

	def __getitem__(self, x):
		if type(x) == int:
			return super(ERgraph, self).__getitem__(x)
		elif type(x) == str:
			return self.lookup[x]
		else:
			raise Exception("Unexpected:  ", x)

	@dispatch(MetaRelation)
	def __contains__(self, x):
		"""
		Function that determines whether a specific MetaRelation.name is present in the graph.

		Parameters
		----------
		x : MetaRelation 
			A MetaRelation object.

		Returns
		-------
		bool (Is x present?)
		"""

		#if type(x) == int:
		#	return super(ERgraph, self).__contains__(x)
		return x in self


	@dispatch(str)
	def __contains__(self, x):
		"""
		Function that determines whether a specific MetaRelation.name is present in the graph.

		Parameters
		----------
		x : str (MetaRelation name)
			Name of a MetaRelation object.

		Returns
		-------
		bool (Is x present?)
		"""

		#if type(x) == int:
		#	return super(ERgraph, self).__contains__(x)
		if type(x) == str:
			return x in self.lookup
		else:
			raise Exception("Unexpected: ", x)

	def __str__(self):
		"""
		Function that expresses the ERgraph as string 

		Parameters
		----------
		
		Returns
		-------
		"""

		s = "ERgraph:\n-Name:%s\n" % self.name
		for r in self.graph:
			s+= "\tMetaRel:%s, dom1:%d, dom2:%d, arity:%d\n" % (r["name"], r["lenDomain1"], r["lenDomain2"], r["arity"])
		return s

