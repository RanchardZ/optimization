import os, sys
import numpy as np 
from copy import copy
from benchmark import Benchmark, check_shape
from StochasticOptimization_gamma_test.common import CODE_PATH

class Sphere(Benchmark):
	def __init__(self, dim_num = 30, bound = [-100., 100.]):
		super(Sphere, self).__init__(dim_num, bound, False)
		self.global_optimum = np.zeros(dim_num)

	def evaluate(self, population):
		population = check_shape(population)
		return np.sum(population ** 2, axis = 1)

class Schaffer(Benchmark):
	def __init__(self, dim_num = 2, bound = [-100., 100.]):
		super(Schaffer, self).__init__(dim_num, bound, False)
		self.global_optimum = np.zeros(dim_num)

	def evaluate(self, population):
		population = check_shape(population)
		return 0.5 + ((np.sin(np.sqrt(np.sum(population**2, axis = 1))))**2 -0.5) /\
					 ((1. + .001 * (np.sum(population**2, axis = 1)))**2)

class Rastrigin(Benchmark):
	def __init__(self, dim_num = 50, bound = [-5.12, 5.12], A = 10.):
		super(Rastrigin, self).__init__(dim_num, bound, False)
		self.global_optimum = np.zeros(dim_num)
		self.A 				= A

	def evaluate(self, population):
		population = check_shape(population)
		return self.A * self.dimensions + np.sum(population**2 - self.A*np.cos(2*np.pi*population), axis = 1)

class Griewank(Benchmark):
	def __init__(self, dim_num = 50, bound = [-600., 600.]):
		super(Griewank, self).__init__(dim_num, bound, False)
		self.global_optimum = np.zeros(dim_num)

	def evaluate(self, population):
		population = check_shape(population)
		return 1. / 4000. * np.sum(population * population, axis = 1) -\
							np.multiply.reduce(np.cos(population / np.sqrt((np.arange(self.dimensions) + 1.))), axis = 1) + 1.

class Rosenbrock(Benchmark):
	def __init__(self, dim_num = 50, bound = [-50., 50.]):
		super(Rosenbrock, self).__init__(dim_num, bound, False)
		self.global_optimum = np.zeros(dim_num)
	def evaluate(self, population):
		population = check_shape(population)
		return np.sum(100. * (population[:, : -1]**2 - population[:, 1:])**2 + (population[:, : -1] - 1)**2, axis = 1)



class Ackley(Benchmark):
	def __init__(self, dim_num = 200, bound = [-32.768, 32.768]):
		super(Ackley, self).__init__(dim_num, bound, False)
		self.global_optimum = np.zeros(dim_num)

	def evaluate(self, population):
		population = check_shape(population)
		m, n = population.shape
		tmp = np.sum((population / np.sqrt(n))**2, axis = 1)
		return 20. - 20. * np.exp(-0.2 * np.sqrt(tmp)) - np.exp(np.sum(np.cos(2 * np.pi * population), axis = 1) / n) + np.e