	
import torch as t
from torch.autograd import Variable
import os

class NXmodelProto(t.nn.Module):

	""" This class is the father of the pytorch modules used in the ER datafusion
	wrapper. It implements basic functions, leaving only the init and the forward empty"""

	def __init__(self):
		super(NXmodelProto, self).__init__()

	def getWeights(self):
		return self.state_dict()

	def getParameters(self):
		return list(self.parameters())

	def init_weights(self, m):
		if isinstance(m, t.nn.Conv1d) or isinstance(m, t.nn.Linear) or isinstance(m, t.nn.Bilinear):
			print ("Initializing weights...", m.__class__.__name__)
			t.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)
		elif isinstance(m, t.nn.Embedding):
			print ("Initializing weights...", m.__class__.__name__)
			t.nn.init.xavier_uniform_(m.weight)
		
	def getNumParams(self):
		p=[]
		for i in self.parameters():
			p+= list(i.data.cpu().numpy().flat)
		print ('Number of parameters=',len(p))	

