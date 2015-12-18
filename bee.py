import random, sys
import numpy as np 
from copy import copy 
from species import Species 
from util import printMessage, rouletteWheelSelection

class ABC(Species):

	def __init__(self, pop_num, dim_num, target, **kwargs):
		super(ABC, self).__init__(pop_num, dim_num, target, 'ABC', **kwargs)
		self.limit_of_exploitation 	= self.pop_num * self.dim_num
		self.trials 				= [0] * self.pop_num

	def oneIteration(self):
		eval_inc = 0
		if self.terminator.n_iter % 2 == 0:
			eval_inc += self.sendEmployedBees()
		else:
			eval_inc += self.sendOnlookerBees()
		eval_inc += self.sendScoutBees()
		self.findBest()
		return eval_inc

	def sendEmployedBees(self):
		param2change = np.random.randint(low = 0, high = self.dim_num, size = self.pop_num)
		for idx, (feat, value) in enumerate(zip(self.population, self.values)):
			neighbor_idx = np.random.randint(low = 0, high = self.pop_num)
			while neighbor_idx == idx:
				neighbor_idx = np.random.randint(low = 0, high = self.pop_num)
			neighbor = self.population[neighbor_idx]

			new_feat = copy(feat)
			new_feat[param2change[idx]] = feat[param2change[idx]] + \
										 (feat[param2change[idx]] - \
										  neighbor[param2change[idx]]) * \
										 (np.random.rand() - 0.5) * 2
			new_feat = self.optimization_target.bounder(new_feat)
			new_value = self.optimization_target.evaluate(new_feat)

			if new_value < value:
				self.population[idx] 	= new_feat
				self.values[idx] 		= new_value
				self.trials[idx]	    = 0
			else:
				self.trials[idx]	   += 1

		return self.pop_num

	def sendOnlookerBees(self):
		param2change = np.random.randint(low = 0, high = self.dim_num, size = self.pop_num)
		fitnesses = calculateFitnesses(self.values)
		probs = calculateProbs(fitnesses)

		for _ in range(self.pop_num):
			select_idx = rouletteWheelSelection(probs)
			neighbor_idx = np.random.randint(low = 0, high = self.pop_num)
			while neighbor_idx == select_idx:
				neighbor_idx = np.random.randint(low = 0, high = self.pop_num)
			neighbor = self.population[neighbor_idx]
			feat = self.population[select_idx]
			new_feat = copy(feat)
			new_feat[param2change[select_idx]] = feat[param2change[select_idx]] + \
												(feat[param2change[select_idx]] - \
												 neighbor[param2change[select_idx]]) * \
												(np.random.rand() - 0.5) * 2
			new_feat = self.optimization_target.bounder(new_feat)
			new_value = self.optimization_target.evaluate(new_feat)
			if new_value < self.values[select_idx]:
				self.population[select_idx] = new_feat
				self.values[select_idx] = new_value
				self.trials[select_idx] = 0
			else:
				self.trials[select_idx] += 1
		return self.pop_num

	def sendScoutBees(self):
		max_idx = np.argmax(self.trials)
		if self.trials[max_idx] >= self.limit_of_exploitation:
			feat = self.optimization_target.generateFeat()
			value = self.optimization_target.evaluate(feat)
			self.population[max_idx] = feat
			self.values[max_idx] = value
			self.trials[max_idx] = 0
			return 1
		else:
			return 0

class SLG(Species):
	""" A different implementation of Artificial Bee Algorithm (ABC)
	"""
	def __init__(self, pop_num, dim_num, target, **kwargs):
		super(SLG, self).__init__(pop_num, dim_num, target, 'SLG', **kwargs)
		self.p_local_search = 0.5

	def oneIteration(self):
		eval_inc = 0
		eval_inc += self.move()
		self.findBest()
		return eval_inc

	def move(self):
		param2change = np.random.randint(low = 0, high = self.dim_num, size = self.pop_num)
		fitnesses = calculateFitnesses(self.values)
		probs = calculateProbs(fitnesses)

		for _ in range(self.pop_num):
			if np.random.rand() < self.p_local_search:
				select_idx = rouletteWheelSelection(probs)
			else:
				select_idx = np.random.randint(low = 0, high = self.pop_num)

			neighbor_idx = np.random.randint(low = 0, high = self.pop_num)
			while neighbor_idx == select_idx:
				neighbor_idx = np.random.randint(low = 0, high = self.pop_num)
			neighbor = self.population[neighbor_idx]
			feat = self.population[select_idx]
			new_feat = copy(feat)
			new_feat[param2change[select_idx]] = feat[param2change[select_idx]] + \
												(feat[param2change[select_idx]] - \
												 neighbor[param2change[select_idx]]) * \
												(np.random.rand() - 0.5) * 2
			new_feat = self.optimization_target.bounder(new_feat)
			new_value = self.optimization_target.evaluate(new_feat)
			if new_value < self.values[select_idx]:
				self.population[select_idx] = new_feat
				self.values[select_idx] = new_value
		return self.pop_num

def calculateFitnesses(values):
	pos_mask = values >= 0
	neg_mask = values < 0
	return 1. * pos_mask  / (values + 1.) +\
		  (1. - values) * neg_mask

def calculateProbs(populationFitnesses):
	maxFit = np.max(populationFitnesses)
	return .9 * populationFitnesses / maxFit + 0.1