import os, sys
import numpy as np
from copy import copy, deepcopy
from individual import Individual
from benchmark_funcs.cec_05_real_parameter import *
from benchmark_funcs.cec_13_real_parameter import *
from benchmark_funcs.cec_13_lsop import *
from benchmark_funcs.cec_14_real_parameter import *
from benchmark_funcs.karaboga import *
from terminator import Terminator
from archiver import Archiver
from common import ROOT_PATH
from util import printMessage

class Species(object):
	''' Species corresponds to population-based metaheuristics'''
	def __init__(self, pop_num, dim_num, target, species_name, **kwargs):
		self.pop_num, self.dim_num 		= pop_num, dim_num
		self.target, self.species_name 	= target, species_name		
		self._kwargs = kwargs
		
		self.init_terminator(self._kwargs)
		self.init_target()
		self.init_archiver(self._kwargs)
		self.init_population()
		self.init_info(self._kwargs)
		self.init_params()

	def evolveOneIter(self):
		if not self.terminator.should_terminate():
			eval_inc 		= self.oneIteration()
			self.terminator.add_stat(iter_inc = 1, eval_inc = eval_inc)
			self.terminator.update_diversity(get_diversity(self.population))
			self.observe(eval_inc)
			self.archive()
		self.clean()

	def evolveSpecies(self):

		while not self.terminator.should_terminate():
			eval_inc 		= self.oneIteration()
			self.terminator.add_stat(iter_inc = 1, eval_inc = eval_inc)
			self.terminator.update_diversity(get_diversity(self.population))
			self.observe(eval_inc)
			self.archive()
		self.clean()

	def oneIteration(self):
		"""This function should be overridden to perform one iteration and return the number of evaluations 
		"""
		raise NotImplementedError

	def observe(self, eval_inc):
		if self.info_switch:
			if self.point_switch:
				if (self.terminator.t_type == 'evaluation_terminator' and abs(self.terminator.n_eval - self.cur_point) < eval_inc) or\
				   (self.terminator.t_type == 'iteration_terminator' and self.terminator.n_iter == self.cur_point):
					msg, src = self.terminator.get_message()
					self.print_info(msg, src)
					try:
						print self.techs_off
					except:
						pass
					try:
						self.cur_point = self.observe_points.next()
					except:
						pass
			else:
				msg, src = self.terminator.get_message()
				self.print_info(msg, src)
		else:
			return

	def archive(self):
		n_iter, n_eval, diversity = self.terminator.get_stat()
		content = [n_iter, n_eval, self.best_ind.value, self.global_best_ind.value, diversity]
		self.archiver.writeToFile(content)

	def clean(self):
		""" This function performs some cleaning work
		"""
		self.archiver.closeFile()

	def individualize(self, feature, value):
		return Individual(feature, value)

	def find_best(self):
		idx_min 		= np.argmin(self.values)
		self.best_ind 	= self.individualize(self.population[idx_min], self.values[idx_min])
		try:
			if self.best_ind > self.global_best_ind:
				self.global_best_ind = copy(self.best_ind)
		except:
			self.global_best_ind = copy(self.best_ind)

	def print_info(self, msg, src):
		content = '{0} : <bestValue: {1:^10.5E}>, <gbestValue: {2:^10.5E}>'.format(msg, self.best_ind.value, self.global_best_ind.value)
		printMessage(src, 'INFO', content)

	def init_terminator(self, kwargs):
		t_type 	= kwargs.setdefault('terminator_type', 'evaluation_terminator')
		self.terminator = Terminator(max_iterations 	= kwargs.setdefault('max_iterations', 1E3),
									 max_evaluations 	= kwargs.setdefault('max_evaluations', 5E4),
									 terminator_type 	= t_type,
									 source 			= self.species_name)

	def init_target(self):
		#exec("self.optimization_target = %s(kwargs.get('seed'))" % (target))
		exec("self.optimization_target = %s(%d)" % (self.target, self.dim_num))
		#self.optimization_target 	= target

	def init_archiver(self, kwargs):
		self.epoch  			= kwargs.setdefault('epoch', 1)
		archive_switch 			= kwargs.setdefault('archive_switch', False)
		self.archiver 			= Archiver(self.species_name,\
								 os.path.join(ROOT_PATH, kwargs.setdefault('project_name', 'test'), kwargs.setdefault('experiment_name', 'testSpecies'), 'rawData'),\
								 '_'.join([self.species_name, str(self.pop_num), str(self.dim_num), self.target, str(int(self.terminator.max_eval)), str(self.epoch)]) + '.sto', archive_switch)
		self.archiver.openFileToWrite()

	def init_population(self):
		self.population 		= self.optimization_target.generatePopulation(self.pop_num)
		self.values 			= self.optimization_target.evaluate(self.population)
		self.find_best()
		self.terminator.add_stat(eval_inc = self.pop_num)
		self.terminator.update_diversity(get_diversity(self.population))
		self.archive()

	def init_info(self, kwargs):
		# Information output during processing
		self.info_switch 		= kwargs.setdefault('info_switch', True)
		self.point_switch 		= kwargs.setdefault('point_switch', True)
		self.observe_points 	= iter(kwargs.setdefault('observe_points', [5E04, 1.2E05, 3E05, 5E05, 10E05, 15E05, 20E05, 25E05, 30E05]))
		self.cur_point 			= self.observe_points.next()

	def init_params(self):
		pass




def get_diversity(population):
	""" any kind of definition for diversity can be implemented here
	"""
	# average variance of each component
	return np.average(np.var(population, axis = 0))
	return np.average(np.std(population, axis = 0))
	return np.average(np.std(population, axis = 0) / ((self.benchmarks.bound[1] - self.benchmarks.bound[0]) / 2))
