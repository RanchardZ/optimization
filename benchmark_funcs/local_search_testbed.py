import os, sys
import numpy as np 
from copy import copy
from StochasticOptimization_gamma_test.common import CODE_PATH
from benchmark import Benchmark


class obj32(object):
	def evaluate(self, feat):
		return 100. * ((feat[1] - feat[0]**2)**2) + (1. - feat[0])**2

class sphere_2d(object):
	def evaluate(self, feat):
		return feat[0]**2 + feat[1]**2