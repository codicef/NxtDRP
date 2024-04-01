
import pickle as cPickle
import numpy as np
from multipledispatch import dispatch
import time
import NXTfusion.NXTfusion as NX #import Entity
from scipy.sparse import coo_matrix

class SideInfo(object):

	"""
	Class that encapsulated the side information raw data in order to be efficiently processed by NXTfusion. You can use this class to wrap side information vectors analogously to how DataMatrix wraps matrix/relations.
	"""

	@dispatch(str,  NX.Entity, dict)
	def __init__(self, name, ent1, data):
		"""
			One of the alternative constructors for the SideInfo class.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			ent1 : Entity 
				Entity object representing the object on the dimension 0
			data: dict
				Dict containing ent1 objects as keys and feature vectors (side information) as values.

			Returns
			-------
		"""
		self.name = name
		self.ent1 = ent1
		self.dtype = type
		self.data = {} # data = {idx:[features]}
		missing = 0
		l = len(data.values()[0])
		for i in ent1:
			try:
				self.data[ent1[i]] = data[i]
			except:
				self.data[ent1[i]] = [0]*l
				missing += 1
		print ("Missing: ", missing)
		for i in self.data.items():
			assert len(i[1]) == len(self.data.values()[0])

	@dispatch(str,  NX.Entity, np.ndarray)
	def __init__(self, name, ent1, data):
		"""
			One of the alternative constructors for the SideInfo class.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			ent1 : Entity 
				Entity object representing the object on the dimension 0
			data: numpy.ndarray
				Numpy array that contains the side information. It has shape (ent1 obj, feature length), similarly to a scikit-learn feature vector.

			Returns
			-------
		"""
		self.name = name
		self.ent1 = ent1
		self.dtype = type
		self.data = {} # data = {idx:[features]}
		missing = 0
		assert len(ent1) == data.shape[0], "ERROR: data.shape[0] and len(ent1) do not match. You should provide one side info vector for each object in the entity."
		l = data.shape[1]
		for i in ent1:
			try:
				self.data[ent1[i]] = data[i]
			except:
				self.data[ent1[i]] = [0]*l
				missing += 1
		print ("Missing: ", missing)

	@dispatch(str,  NX.Entity, coo_matrix)
	def __init__(self, name, ent1, data):
		"""
			One of the alternative constructors for the SideInfo class.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			ent1 : Entity 
				Entity object representing the object on the dimension 0
			data: scipy.sparse.coo_matrix
				Scipy coo_matrix that contains the side information. It has shape (ent1 obj, feature length), similarly to a scikit-learn feature vector. It can be sparse, but currently the sparsity during mini batching is NOT supported.

			Returns
			-------
		"""
		self.name = name
		self.ent1 = ent1
		self.dtype = type
		d = {}
		l = data.shape[1]
		data = data.toarray()
		print(data.shape)
		i = 0
		while i < len(data.shape[1]):
			d[i] = data[i,:]
		data = d
		self.data = {} # data = {idx:[features]}
		missing = 0
		for i in ent1:
			print (i, ent1[i], data[i])
			try:
				self.data[ent1[i]] = data[i][:100]
			except:
				self.data[ent1[i]] = [0]*100
				missing += 1
		print ("Missing: ", missing)
		#for i in self.data.items():
		#	assert len(i[1]) == len(self.data.values()[0])

	@dispatch( str)
	def __init__(self, path):
		"""
		This constructor reads a serialized (SideInfo.dump()) SideInfo object.			
			Parameters
			----------
			name: str
				Path to the serialized SideInfo object.

			Returns
			-------
		"""

		print( "Loading %s..." % path)
		start = time.time()
		store = cPickle.load(open(path))
		stop = time.time()

		try:
			store["name"]
			store["ent1"]
			store["data"]
		except:
			print( "ERROR: wrong format, check file content")
			exit(1)
		self.name= store["name"]
		self.ent1 = store["ent1"]
		self.data = store["data"]
		print ("Done in %.2fs." % (stop-start))

	def normalize(self):
		"""
		Method that standardizes the matrix with the formula x' = (x - mu)/s, where mu is the mean and s is the standard deviation.

		Returns
		-------
		None
		"""

		l = []
		for i in self.data.values():
			l += list(i[1])
		#print l
		print( len(l))
		mu = np.mean(l)
		s = np.std(l)
		print ("mu = %f, s= %f" % (mu, s))
		for i in self.data.items():
			self.data[i[0]] = [i[1][0], (i[1][1]-mu)/s]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, x):
		return self.data[x]

	def dump(self, path=None):
		"""
		Method that serializes the SideInfo storing it at the selected path.

		path: str
			Destination path for the serialized file

		Returns
		-------
		None
		"""
		if path == None:
			print ("Storing...")
			start = time.time()
			store = {"name":self.name, "ent1":self.ent1, "data":self.data}
			cPickle.dump(store, open("marshalled/"+self.name+".side.nx", "w"))
			stop = time.time()
			print ("Done in %.2fs." % (stop-start))
		else:
			print ("Storing...")
			start = time.time()
			store = {"name":self.name, "ent1":self.ent1, "data":self.data}
			cPickle.dump(store, open(path, "w"))
			stop = time.time()
			print ("Done in %.2fs." % (stop-start))


class DataMatrix(object):

	"""
	The input "data" format should be: {(ent1, ent2): value} for all the observed elements in the matrix.

	The format in which the data is stored in the DataMatrix object is the following: 
	featsHT = {domain1Name_numeric : [ numpy16_domain2Names_numeric, numpyX_labels ]}

	"""

	@dispatch(str, NX.Entity,  NX.Entity, dict, type)
	def __init__(self, name:str, ent1:NX.Entity, ent2:NX.Entity, data:dict, dtype:type) :
		"""One of the alternative constructors for the DataMatrix class.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			ent1 : Entity 
				Entity object representing the object on the dimension 0
			ent2: Entity
				Entity object representing the object on the dimension 1
			data: dict  {(ent1, ent2): value}
				Hash table containing the (sparse) elements and in the matrix describing the relation. The input "data" format should be: {(ent1, ent2): value} for all the observed elements in the matrix.
			dtype: numpy.dtype
				The smallest possible type that could be used to store the elements of the matrix (e.g. np.int16)

			Returns
			-------
		"""
		self.name = name
		self.ent1 = ent1
		self.ent2 = ent2
		self.dtype = type
		print( "Building features for matrix %s..." % name)
		self.data, self.size = buildPytorchFeatsHT(data, ent1, ent2, dtype)
		#print (list(self.data.items())[:3])
		print ("Size: ", self.size)

	@dispatch(str, NX.Entity,  NX.Entity, coo_matrix)
	def __init__(self, name:str, ent1:NX.Entity, ent2:NX.Entity, data:coo_matrix) :
		"""One of the alternative constructors for the DataMatrix class.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			ent1 : Entity 
				Entity object representing the object on the dimension 0
			ent2: Entity
				Entity object representing the object on the dimension 1
			data: coo_matrix
				scipy.sparse.coo_matrix containing the sparse elements and in the matrix describing the relation.

			Returns
			-------
		"""
		self.name = name
		self.ent1 = ent1
		self.ent2 = ent2
		self.dtype = type
		print( "Building features for matrix %s..." % name)
		d = {}
		data = data.todok()
		for i in data.items():
			d[i[0]] = i[1]
		self.data, self.size = buildPytorchFeatsHT(d, ent1, ent2, data.dtype)
		#print (list(self.data.items())[:3])
		print ("Size: ", self.size)


	@dispatch(str, NX.Entity, NX.Entity, np.ndarray)
	def __init__(self, name:str, ent1:NX.Entity, ent2:NX.Entity, data:np.ndarray) :
		"""
			One of the alternative constructors for the DataMatrix class.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			ent1 : Entity 
				Entity object representing the object on the dimension 0
			ent2: Entity
				Entity object representing the object on the dimension 1
			data: numpy.ndarray (matrix)
				Numpy matrix containing the (dense) describing the relation between ent1 and en2. 

			Returns
			-------
				DataMatrix object
		"""
		self.name = name
		self.ent1 = ent1
		self.ent2 = ent2
		assert len(ent1) == data.shape[0], "ERROR: ent1 len is "+str(len(ent1))+" but data.shape[0] is"+str(data.shape[0])
		assert len(ent2) == data.shape[1], "ERROR: ent2 len is "+str(len(ent2))+" but data.shape[0] is"+str(data.shape[0])

		print( "Building features for matrix %s..." % name)
		self.data, self.size = buildPytorchFeatsHTfromNumpy(data, self.ent1, self.ent2)
		#print (list(self.data.items())[:3])
		print ("Size: ", self.size)

	@dispatch(str, np.ndarray)
	def __init__(self, name:str, data:np.ndarray, dtype:type) :
		"""Simplest possible constructor for the DataMatrix class. Entities are inferred.
			
			Parameters
			----------
			name: str
				Name of the data matrix
			data: numpy.ndarray (matrix)
				Numpy matrix containing the (dense) describing the relation between ent1 and en2. 

			Returns
			-------
		"""

		self.name = name
		self.ent1 = NX.Entity(self.name+"_0", list(range(data.shape[0])), dtype = dtype)	
		self.ent2 = NX.Entity(self.name+"_1", list(range(data.shape[1])), dtype = dtype)
		print( "Building features for matrix %s..." % name)
		self.data, self.size = buildPytorchFeatsHTfromNumpy(data, self.ent1, self.ent2)
		#print (list(self.data.items())[:3])
		print ("Size: ", self.size)

	def size(self):
		"""
		Function that return the size of the relation (number of elements in the matrix).

		Returns
		-------
			Size of the relation in the DataMatrix object
		"""
		size = 0
		for i in data.items():
			size += len(i[1][0])
		print ("Size: ", size)
		return size

	@dispatch( str)
	def __init__(self, path):
		"""Constructor that reads the DataMatrix from a previously serialized DataMatrix object.
			
			Parameters
			----------
			path: str
				Path of the serialized DataMatrix
			
			Returns
			-------
					"""

		print( "Loading %s..." % path)
		start = time.time()
		store = cPickle.load(open(path))
		stop = time.time()

		try:
			store["name"]
			store["ent1"]
			store["ent2"]
			store["data"]
		except:
			print ("ERROR: wrong format, check file content")
			exit(1)
		self.name= store["name"]
		self.ent1 = store["ent1"]
		self.ent2 = store["ent2"]
		self.data = store["data"]
		print ("Done in %.2fs." % (stop-start))

	def standardize(self):
		"""
		Method that standardizes the matrix with the formula x' = (x - mu)/s, where mu is the mean and s is the standard deviation.

		Returns
		-------
		None
		"""
		l = []
		for i in self.data.values():
			l += list(i[1])
		#print l[:100]
		#print len(l)
		mu = np.mean(l)
		s = np.std(l)
		print ("mu = %f, s= %f" % (mu, s))
		tmp = []
		for i in self.data.items():
			res = (i[1][1]-mu)/s
			#print res
			#raw_input()
			self.data[i[0]] = [i[1][0], res]
			tmp += res.tolist()
		#print len(tmp)
		#print tmp[:100]
		tmu = np.mean(tmp)
		ts = np.std(tmp)
		#print tmu, ts
		assert abs(tmu) < 0.0001 
		assert abs(ts-1) < 0.0001
		
	def toHashTable(self)-> dict:
		"""
		Method that returns an hash table (dict) containing the DataMatrix data.
		
		Returns
		-------
			dict
		"""

		db = {}
		for p1 in self.data.items():
			#print p1
			i = 0
			assert len(p1[1][0]) == len(p1[1][1])
			while i < len(p1[1][0]):
				db[tuple(sorted([self.ent1[p1[0]], self.ent2[int(p1[1][0][i])]]))] = p1[1][1][i]
				i+=1
		print( "Found %d entries" % len(db))
		#print db.items()[:10]
		return db

	def dump(self, path= None):
		if path == None:
			print ("Storing...")
			start = time.time()
			store = {"name":self.name, "ent1":self.ent1, "ent2":self.ent2, "data":self.data}
			cPickle.dump(store, open("marshalled/"+self.name+".nx", "w"))
			stop = time.time()
			print ("Done in %.2fs." % (stop-start))
		else:
			print ("Storing...")
			start = time.time()
			store = {"name":self.name, "ent1":self.ent1, "data":self.data}
			cPickle.dump(store, open(path, "w"))
			stop = time.time()
			print ("Done in %.2fs." % (stop-start))

@dispatch(np.ndarray, NX.Entity, NX.Entity)	
def buildPytorchFeatsHTfromNumpy(data:np.ndarray, domain1:NX.Entity, domain2:NX.Entity) -> (dict, int):
	""" This function produces the data structure that is internally used to pass training data to the wrapper.fit function. This is now transparent to the user.
		
		Parameters
		----------
		data: numpy.ndarray
			Numpy matrix containing the matrix that represents the relation between ent1 and ent2
		domain1 : Entity
			Entity1
		domain2 : Entity
			Entity2

		Returns
		-------
		Dict internally used to feed the data to the NX.Wrapper object.
		The format is the following
		featsHT = {domain1Name_numeric : [ numpy16_domain2Names_numeric, numpyX_labels ]}
	"""

	size = 0
	x = {}
	i = 0
	while i < len(domain1):
		x[i] = [[],[]]
		i+=1
	if domain1 == domain2:
		print( " *** identified as self relation.")
		selfRelation = True
	else:
		print (" *** identified as asymmetric relation.")
		selfRelation = False
	for i in range(0, len(domain1)):
		for j in range(0, len(domain2)):
			if selfRelation:
				try:
					tmp = tuple(sorted([i,j])) #probably dangerous here
				except:
					continue
			else:
				tmp = tuple([i,j])
			x[tmp[0]][0].append(tmp[1])
			x[tmp[0]][1].append(data[i,j])
	xf = {}

	for i in x.items():
		size += len(i[1][0])
		xf[i[0]] = [np.array(i[1][0], dtype=domain1.dtype), np.array(i[1][1], dtype=data.dtype)]
	return xf, size

@dispatch(dict, NX.Entity, NX.Entity, np.dtype)
def buildPytorchFeatsHT(data:dict, domain1:NX.Entity, domain2:NX.Entity, relDtype:type)-> (dict, int):
	""" This function produces the data structure that is internally used to pass training data to the wrapper.fit function. This is now transparent to the user.
		
		Parameters
		----------
		data: dict {(ent1[i], ent2[j]): value[i,j]}

			Dict in the following format: {(ent1[i], ent2[j]): value[i,j]} containing the matrix that represents the relation between ent1 and ent2. This format can be used to input sparse matrices
		domain1 : Entity
			Entity1
		domain2 : Entity
			Entity2
		relDtype : numpy.dtype
			The smallest np.dtype sufficient to represent the values in the matrix. 

		Returns
		-------
		Dict internally used to feed the data to the NX.Wrapper object.
		The format is the following
		featsHT = {domain1Name_numeric : [ numpy16_domain2Names_numeric, numpyX_labels ]}
	
	"""

	size = 0
	x = {}
	i = 0
	while i < len(domain1):
		x[i] = [[],[]]
		i+=1
	if domain1 == domain2:
		print( " *** identified as self relation.")
		selfRelation = True
	else:
		print (" *** identified as asymmetric relation.")
		selfRelation = False
	for i in data.items():
		if selfRelation:
			try:
				tmp = tuple(sorted([domain1[i[0][0]], domain2[i[0][1]]])) #probably dangerous here
			except:
				continue
		else:
			tmp = tuple([domain1[i[0][0]], domain2[i[0][1]]])
		x[tmp[0]][0].append(tmp[1])
		x[tmp[0]][1].append(i[1])
	xf = {}

	for i in x.items():
		size += len(i[1][0])
		xf[i[0]] = [np.array(i[1][0], dtype=domain1.dtype), np.array(i[1][1], dtype=relDtype)]
	return xf, size


