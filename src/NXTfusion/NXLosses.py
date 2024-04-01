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
import time
import torch as t
from torch.autograd import Variable

class FocalLoss(t.nn.Module):
	"""
	Implementation of the FocalLoss, which is used as loss for heavily unbalanced binary predictions. It is bult as a convetional pytorch module.
	"""
	def __init__(self, alpha=1, gamma=2, logits=True, reduction="sum"):
		"""
			Constructor for the FocalLoss.

			Parameters
			----------
			alpha : int
				Parameter of the loss
			gamma : int
				Parameter of the loss
			logits : bool
				Uses logits if True
		"""
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.logits = logits
		self.reduction = reduction

	def forward(self, inputs, targets):
		"""
		:meta private:
		"""

		if self.logits:
			BCE_loss = t.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
		else:
			BCE_loss = t.nn.functional.binary_cross_entropy(inputs, targets, reduce=False)
		pt = t.exp(-BCE_loss)
		F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

		if self.reduction == "mean":
			return t.mean(F_loss)
		elif self.reduction == "sum":
			return t.sum(F_loss)
		else:
			return F_loss



class LossWrapper(t.nn.Module):

	"""
	Class that wraps any pytorch loss allowing for ignore index. In the Matrix Factorization context it may be useful to define a value indicating missing values even when performing a regression, for example if the goal is to predict a sparsely observed real-valued matrix.
	"""

	def __init__(self, loss:t.nn.Module, type:str, ignore_index:int):
		"""
		Constructor for the wrapper.

		Parameters
		----------
		loss : t.nn.Module
			The argument can be any pytorch compatible loss functioni
		type:
			Specifies wheter is a regression or a binay prediction (deprecate?)
		ignore_index : int
			Specifies which value should be ignored while computing the loss, to allow for the presence of missing values in the matrix/relation.

		Returns
		-------
		"""
		super(LossWrapper, self).__init__()
		self.loss = loss
		self.ignore_index = ignore_index
		self.type = type
	
	def __call__(self, input, target):
		"""
		Function defining the forward pass for this wrapper. It implements the ignore_index filtering and then it calls the actual self.loss on the remaining values.

		Parameters
		----------
		input : t.nn.Tensor
			Pytorch tensor containing the predicted values
		target : t.nn.Tensor
			Pytorch tensor containing the target values

		Returns
		-------
		Loss score computed only for the target values that are not equal to self.ignore_index.
		"""
		input = input.view(-1)
		target = target.view(-1)
		if self.ignore_index != None:
			mask = target.ne(self.ignore_index)
			input = t.masked_select(input, mask)
			target = t.masked_select(target, mask)
		#t2 = time.time()
		
		r = self.loss(input, target)
		#t3 = time.time()
		#print t2-t1, t3-t2
		#raw_input()
		return r

