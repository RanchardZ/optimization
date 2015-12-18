import os, sys
import numpy as np 
from copy import copy
#from common import CODE_PATH
CODE_PATH = '/home/hxx/Desktop/hehe/zhCode/StochasticOptimization_gamma_test/'
class Benchmark(object):
	"""Defines a global optimization benchmark problem
	"""
	def __init__(self, dim_num, bound, maximize = False):
		self.dim_num 	= dim_num
		self.bound 		= bound
		self.maximize 	= maximize
	def bounder(self, feature, method = 0, parent_feature = None):
		"""also work for groups of feature (namely population)
		"""
		feature = check_shape(feature)
		if method == 0 or method == 'reinitializing':
			####### reinitializing bounding strategy #######
			pop_num, dim_num 	= feature.shape
			#overflowTable 		= feature > self.bound[1] * np.ones(self.dim_num)
			#underflowTable 		= feature < self.bound[0] * np.ones(self.dim_num)
			overflowTable 		= feature > self.bound[1]
			underflowTable 		= feature < self.bound[0]
			return (np.random.rand(pop_num, dim_num) * (self.bound[1] - self.bound[0]) + self.bound[0]) * (overflowTable + underflowTable) +\
			            feature * (1 - overflowTable - underflowTable)
		elif method == 1 or method == 'bounceback':
			####### bounce back strategy #############
			overflowTable 		= feature > self.bound[1] * np.ones(self.dim_num)
			underflowTable 		= feature < self.bound[0] * np.ones(self.dim_num)
			return (2 * self.bound[1] * np.ones(self.dim_num) - feature) * overflowTable +\
				   (2 * self.bound[0] * np.ones(self.dim_num) - feature) * underflowTable +\
				   feature * (1 - overflowTable - underflowTable)
		elif method == 2 or method == 'atbounds':
			####### bounded at the ends ##############
			return np.maximum(np.minimum(feature, self.bound[1] * np.ones(self.dim_num)), self.bound[0] * np.ones(self.dim_num))
		elif (method == 3 or method == 'middle') and (not (target_features is None)):

			overflowTable 		= feature > self.bound[1] * np.ones(self.dim_num)
			underflowTable 		= feature < self.bound[0] * np.ones(self.dim_num)
			return (parent_feature + self.bound[1]) / 2 * overflowTable +\
				   (parent_feature + self.bound[0]) / 2 * underflowTable +\
				   (feature) * (1 - overflowTable - underflowTable)
		else:
			raise NotImplementedError
		
	def evaluate(self, population):
		raise NotImplementedError

	def generateFeat(self):
		return np.random.rand(self.dim_num) * (self.bound[1] - self.bound[0]) + self.bound[0]

	def generatePopulation(self, popSize):
		return np.random.rand(popSize, self.dim_num) * (self.bound[1] - self.bound[0]) + self.bound[0]

	def __str__(self):
		return '%s (%d dim_num)' % (self.__class__.__name__, self.dim_num)

	def __repr__(self):
		return self.__str__()

	def __call__(self, population):
		values = self.evaluate(population, {})
		return values

def check_shape(population):
	""" Turn a one-dimensional numpy array into a two-dimensional one.
	"""
	if len(population.shape) == 1:
		return population.reshape(1, population.shape[0])
	return population
