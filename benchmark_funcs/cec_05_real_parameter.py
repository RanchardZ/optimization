import os, sys
import numpy as np 
from copy import copy
from constant import CODE_PATH
from benchmark import Benchmark, check_shape


input_data_path 	= os.path.join(CODE_PATH, 'benchmark_funcs', 'CEC_2005_RP_Data')


###########################################################################################################
#         Benchmark Functions for the CEC'2013 Special Session on Real-Parameter Optimization             #
###########################################################################################################

class cec_05_rp_benchmark(Benchmark):
	def __init__(self, dim_num, bound, maximize = False):
		super(cec_05_rp_benchmark, self).__init__(dim_num, bound, maximize)
		self.init_params()

	def init_params(self):
		raise NotImplementedError

	@staticmethod
	def sphere(population):
		population 	= check_shape(population)
		return np.sum(population**2, axis = 1)

	@staticmethod
	def sphere_noise(population):
		population 	= check_shape(population)
		r 			= np.sum(population**2, axis = 1)
		return r * (0.1 * np.abs(np.random.randn()) +1.0)

	@staticmethod
	def schwefel_102(population):
		population 	= check_shape(population)
		ps, D 		= population.shape 
		fit 		= 0.
		tmp_sum 	= 0.
		for i in range(D):
			tmp_sum += population[:, i]
			fit 	+= tmp_sum**2
		return fit

	@staticmethod
	def rosenbrock(population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= np.sum(100. * (population[:, : -1]**2 - population[:, 1:])**2 + (population[:, : -1] - 1.)**2, axis = 1)
		return fit

	@staticmethod
	def F2(x, y):
		fit 		= 100. * ((x**2 - y)**2) + (x - 1.)**2
		return fit

	@staticmethod
	def griewank(population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= 1. / 4000. * np.sum(population * population, axis = 1) -\
						np.multiply.reduce(np.cos(population / np.sqrt((np.arange(D) + 1.))), axis = 1) + 1.
		return fit

	@staticmethod
	def F8(x):
		fit 		= x**2 / 4000. - np.cos(x) + 1.
		return fit

	@classmethod
	def F8F2(cls, population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= np.zeros(ps)
		for i in range(1, D):
			fit += cls.F8(cls.F2(population[:, i-1], population[:, i]))
		fit += cls.F8(cls.F2(population[:, D-1], population[:, 0]))
		return fit

	@staticmethod
	def ackley(population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= np.sum((population/np.sqrt(D))**2, axis = 1)
		fit = 20. - 20.*np.exp(-0.2 * np.sqrt(fit )) - np.exp(np.sum(np.cos(2 * np.pi * population), axis = 1) / D) + np.e
		return fit

	@staticmethod
	def rastrigin(population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= 10. * (D - np.sum(np.cos(2 * np.pi * population), axis = 1)) +\
		        	  np.sum(population**2, axis = 1)
		return fit

	@staticmethod
	def weierstrass(population, a = 0.5, b = 3.0, kmax = 20):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= np.zeros(ps)
		for i, feat in enumerate(population):
			result 	= .0
			for k in range(kmax+1):
				result += (a**k) * (np.cos(2 * np.pi * (b**k) * (feat + 0.5)) - np.cos(2 * np.pi * (b**k) * 0.5))
			fit[i] 	= np.sum(result)
		return fit

	@classmethod
	def high_conditioned_elliptic(cls, population):
		fit 		= cls.elliptic(population)
		return fit

	@staticmethod
	def SchafferF6(x, y):
		tmp1 = x**2 + y**2
		tmp2 = np.sin(np.sqrt(tmp1))
		tmp3 = 1. + 0.001 * tmp1
		fit  = 0.5 + ((tmp2**2 - 0.5)/(tmp3**2))
		return fit

	@classmethod
	def EScafferF6(cls, population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= np.zeros(ps)
		for i in range(1, D):
			fit += cls.SchafferF6(population[:, i-1], population[:, i])
		fit += cls.SchafferF6(population[:, D-1], population[:, 0])
		return fit

	@staticmethod
	def elliptic(population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		constant 	= np.power(1E06, 1.0 / (D - 1.))
		fit 		= np.sum(np.power(constant, np.arange(D)) * (population**2), axis = 1)
		return fit

	def hybrid_composition(self, x):
		z = np.zeros(shape = (self.num_func, self.dim_num))
		# get the raw weights
		w = np.zeros(self.num_func)
		for i in range(self.num_func):
			z[i] 	= x - self.m_o[i]
			sum_sqr = np.sum(z[i]**2)
			w[i] 	= np.exp(-1. * sum_sqr / (2. * self.dim_num * (self.m_sigma[i]**2)))
		w_max = np.max(w)
		coef  = 1. - np.power(w_max, 10.)
		for i in range(self.num_func):
			if w[i] != w_max:
				w[i] *= coef
		## additional code for avoid zero sum. added by Zhenghe##
		w_sum = np.sum(w)
		if w_sum == 0.:
			w += 1E-10
			w_sum = np.sum(w)
		#########################################################
		w = w / np.sum(w)

		fit = 0.0
		z = z / (self.m_lambda.reshape(self.num_func, 1))
		for i in range(self.num_func):
			zm 	= np.dot(z[i], self.m_M[i])
			fit += w[i] * (self.C * self.basic_func(i, zm) / self.m_fmax[i] + self.m_func_biases[i])
		return fit

	# Round Function
	@staticmethod
	def myRound(x):
		return np.sign(x) * np.round(np.abs(x))

	@classmethod
	def myXRound(cls, x, o):
		mask = np.abs(x - o) < 0.5
		return x * mask + cls.myRound(2. * x) / 2. * (1 - mask)

	@classmethod
	def myXRound_s(cls, x):
		mask = np.abs(x) < 0.5
		return x * mask + cls.myRound(2. * x) / 2. * (1 - mask)

	@classmethod
	def EScafferF6NonCont(cls, population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		fit 		= np.zeros(ps)
		for i in range(1, D):
			fit += cls.SchafferF6(cls.myXRound_s(population[:, i-1]), cls.myXRound_s(population[:, i]))
		fit += cls.SchafferF6(cls.myXRound_s(population[:, D-1]), cls.myXRound_s(population[:, 0]))
		return fit

	@classmethod
	def rastriginNonCont(cls, population):
		population 	= check_shape(population)
		ps, D 		= population.shape
		for feat in population:
			feat[:] = map(cls.myXRound_s, feat)
		fit 		= 10. * (D - np.sum(np.cos(2 * np.pi * population), axis = 1)) +\
		        	  np.sum(population**2, axis = 1)
		return fit

	# utility function for loading data
	@staticmethod
	def loadRowVectorFromFile(file_name, col_num):
		with open(file_name, 'r') as data_file:
			return np.array(map(float, data_file.readline().strip().split())[:col_num])


	@staticmethod
	def loadColVectorFromFile(file_name, rows):
		pass

	@staticmethod
	def loadNMatrixFromFile(file_name, N, rows, cols):
		M = np.zeros(shape = (N * rows, cols))
		with open(file_name, 'r') as data_file:
			for i, line in enumerate(data_file.readlines()):
				M[i] = map(float, line.strip().split())[: cols]
		return M.reshape(N, rows, cols)

	@staticmethod
	def loadMatrixFromFile(file_name, rows, cols):
		rm = np.zeros(shape = (rows, cols))
		with open(file_name, 'r') as data_file:
			for i, line in enumerate(data_file.readlines()):
				rm[i] = map(float, line.strip().split())[: cols]
				if i == rows - 1:
					break
		return rm

	def find_m_fax(self):
		self.m_fmax 			= np.zeros(self.num_func)
		for i in range(self.num_func):
			test_point = np.ones(self.dim_num) * 5.0 / self.m_lambda[i]
			test_point_M = np.dot(test_point, self.m_M[i])
			self.m_fmax[i] = np.abs(self.basic_func(i, test_point_M))

	# elementary operations



########################################################
#			    Shifted Sphere Function 			   #
########################################################

class CEC05RP_F01(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F01, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -450.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.sphere(population - self.m_o)
		return fit - 450.

	def init_params(self):
		DEFAULT_FILE_DATA 	= os.path.join(input_data_path, 'sphere_func_data.txt')
		self.m_o 			= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)


########################################################
#            Shifted Schwefel's Function               #
########################################################

class CEC05RP_F02(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F02, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -450.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.schwefel_102(population - self.m_o)
		return fit - 450.

	def init_params(self):
		DEFAULT_FILE_DATA 	= os.path.join(input_data_path, 'schwefel_102_data.txt')
		self.m_o 			= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)

########################################################
#  Shifted Rotated High Conditioned Elliptic Function  #
########################################################

class CEC05RP_F03(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F03, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -450.

	def evaluate(self, population):
		population 	= check_shape(population)
		population 	= np.dot(population - self.m_o, self.m_mat)
		fit 		= self.high_conditioned_elliptic(population)
		return fit - 450.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'high_cond_elliptic_rot_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'elliptic_M_D%d.txt' % self.dim_num)
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)
		self.m_mat 				= self.loadMatrixFromFile(DEFAULT_FILE_MX, self.dim_num, self.dim_num)

########################################################
# Shifted Schwefel's Problem 1.2 with Noise in Fitness #
########################################################

class CEC05RP_F04(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F04, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -450.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.schwefel_102(population - self.m_o)
		fit 		= fit * (np.abs(np.random.randn(population.shape[0])) * 0.4 + 1.0)
		return fit - 450.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'schwefel_102_data.txt')
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)

################################################################
# Shifted Schwefel's Problem 2.6 with Global Optimum on Bounds #
################################################################

class CEC05RP_F05(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F05, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -310.

	def evaluate(self, population):
		population 	= check_shape(population)
		population 	= np.dot(self.m_A, population.T).T
		fit 		= np.max(np.abs(population - self.m_B), axis = 1)
		return fit - 310.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'schwefel_206_data.txt')
		self.m_data 			= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.dim_num + 1, self.dim_num)
		self.m_o 				= np.zeros(self.dim_num)
		self.m_A 				= self.m_data[1:self.dim_num+1, :self.dim_num]
		for i in range(self.dim_num):
			if ((i+1) <= np.ceil(self.dim_num / 4.0)):
				self.m_o[i] = -100.
			elif ((i+1) >= np.floor((3.0 * self.dim_num) / 4.0)):
				self.m_o[i] = 100.
			else:
				self.m_o[i] = self.m_data[0, i]
		self.m_B = np.dot(self.m_A, self.m_o)

########################################################
#			 Shifted Rosenbrock's Function  		   #
########################################################

class CEC05RP_F06(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F06, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = 390.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.rosenbrock(population - self.m_o)
		return fit + 390.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'rosenbrock_func_data.txt')
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num) - 1.0

################################################################
#      Shifted Rotated Griewank's Function without Bounds      #
################################################################

class CEC05RP_F07(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [0., 600.]):
		super(CEC05RP_F07, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -180.

	def bounder(self, feature, method):
		return feature

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.griewank(np.dot(population - self.m_o, self.m_mat))
		return fit - 180.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'griewank_func_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'griewank_M_D%d.txt' % self.dim_num)
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)
		self.m_mat 				= self.loadMatrixFromFile(DEFAULT_FILE_MX, self.dim_num, self.dim_num)

#################################################################
#Shifted Rotated Ackley's Function with Global Optimum on Bounds#
#################################################################

class CEC05RP_F08(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-32., 32.]):
		super(CEC05RP_F08, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -140.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.ackley(np.dot(population - self.m_o, self.m_mat))
		return fit - 140. 

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'ackley_func_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'ackley_M_D%d.txt' % self.dim_num)
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)
		self.m_mat 				= self.loadMatrixFromFile(DEFAULT_FILE_MX, self.dim_num, self.dim_num)
		self.m_o[np.arange(0, self.dim_num, 2)] = -32.

########################################################
#			  Shifted Rastrigin's Function  		   #
########################################################

class CEC05RP_F09(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F09, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -330.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.rastrigin(population - self.m_o)
		return fit - 330.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'rastrigin_func_data.txt')
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)

########################################################
#	     Shifted Rotated Rastrigin's Function  		   #
########################################################

class CEC05RP_F10(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F10, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -330.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.rastrigin(np.dot((population - self.m_o), self.m_mat))
		return fit - 330.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'rastrigin_func_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'rastrigin_M_D%d.txt' % self.dim_num)
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)
		self.m_mat 				= self.loadMatrixFromFile(DEFAULT_FILE_MX, self.dim_num, self.dim_num)

########################################################
#	     Shifted Rotated Weierstrass's Function  	   #
########################################################

class CEC05RP_F11(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-.5, .5]):
		super(CEC05RP_F11, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = 90.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.weierstrass(np.dot((population - self.m_o), self.m_mat))
		return fit + 90.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'weierstrass_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'weierstrass_M_D%d.txt' % self.dim_num)
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)
		self.m_mat 				= self.loadMatrixFromFile(DEFAULT_FILE_MX, self.dim_num, self.dim_num)

########################################################
#				  Schwefel's Problem 2.13  		 	   #
########################################################

class CEC05RP_F12(cec_05_rp_benchmark):
	def  __init__(self, dim_num, bound = [-np.pi, np.pi]):
		super(CEC05RP_F12, self).__init__(dim_num, bound, False)
		self.global_optimum = None
		self.min_fitness = -300.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			self.m_B 	= np.sum(self.m_a * np.sin(feat) + self.m_b * np.cos(feat), axis = 1)
			fit[i] 		= np.sum((self.m_A - self.m_B) ** 2)
		return fit - 460.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'schwefel_213_data.txt')
		self.m_data 			= self.loadMatrixFromFile(DEFAULT_FILE_DATA, 201, self.dim_num)
		self.m_o 				= np.zeros(self.dim_num)
		self.m_a 				= np.zeros(shape = (self.dim_num, self.dim_num))
		self.m_b 				= np.zeros_like(self.m_a)

		self.m_A 				= np.zeros(self.dim_num)
		self.m_B 				= np.zeros(self.dim_num)
		
		for i in range(self.dim_num):
			for j in range(self.dim_num):
				self.m_a[i, j] 	= self.m_data[i, j]
				self.m_b[i, j]	= self.m_data[100+i, j]
			self.m_o[i] = self.m_data[200][i]

		self.m_A = np.sum(self.m_a * np.sin(self.m_o) + self.m_b * np.cos(self.m_o), axis = 1)

#################################################################
#    Shifted Expanded Griewank's plus Rosenbrock's Function     #
#################################################################
		
class CEC05RP_F13(cec_05_rp_benchmark):
	def  __init__(self, dim_num, bound = [-3., 1.]):
		super(CEC05RP_F13, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness 	= -130.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.F8F2(population - self.m_o)
		return fit -130.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'EF8F2_func_data.txt')
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num) - 1.

#################################################################
#        Shifted Rotated Expanded Scaffer's F6 Function         #
#################################################################

class CEC05RP_F14(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-100., 100.]):
		super(CEC05RP_F14, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o
		self.min_fitness = -300.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= self.EScafferF6(np.dot(population - self.m_o, self.m_mat))
		return fit - 300. 

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'E_ScafferF6_func_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'E_ScafferF6_M_D%d.txt' % self.dim_num)
		self.m_o 				= self.loadRowVectorFromFile(DEFAULT_FILE_DATA, self.dim_num)
		self.m_mat 				= self.loadMatrixFromFile(DEFAULT_FILE_MX, self.dim_num, self.dim_num)

#################################################################
#                 Hybrid Composition Function 1                 #
#################################################################

class CEC05RP_F15(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F15, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o[0]
		self.min_fitness = 120.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			fit[i] = self.hybrid_composition(feat)
		return fit + 120.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func1_data.txt')
		self.num_func 			= 10
		self.m_sigma  			= np.ones(self.num_func)
		self.m_lambda 			= np.array([1., 1., 10., 10., 5./60., 5./60., 5./32., 5./32., 5./100., 5./100.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_M 				= np.zeros(shape = (self.num_func, self.dim_num, self.dim_num))
		for i in range(self.num_func):
			for j in range(self.dim_num):
				self.m_M[i, j, j] = 1.
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

	@classmethod
	def basic_func(cls, i, x):
		if i <= 1:
			return cls.rastrigin(x)
		elif i <= 3:
			return cls.weierstrass(x)
		elif i <= 5:
			return cls.griewank(x)
		elif i <= 7:
			return cls.ackley(x)
		elif i <= 9:
			return cls.sphere(x)
		else:
			raise NotImplementedError

#################################################################
#              Rotated Hybrid Composition Function 1            #
#################################################################

class CEC05RP_F16(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F16, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o[0]
		self.min_fitness = 120.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			fit[i] = self.hybrid_composition(feat)
		return fit + 120.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func1_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func1_M_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.ones(self.num_func)
		self.m_lambda 			= np.array([1., 1., 10., 10., 5./60., 5./60., 5./32., 5./32., 5./100., 5./100.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		self.find_m_fax()

	@classmethod
	def basic_func(cls, i, x):
		if i <= 1:
			return cls.rastrigin(x)
		elif i <= 3:
			return cls.weierstrass(x)
		elif i <= 5:
			return cls.griewank(x)
		elif i <= 7:
			return cls.ackley(x)
		elif i <= 9:
			return cls.sphere(x)
		else:
			raise NotImplementedError

#################################################################
#  Rotated Hybrid Composition Function 1 with Noise in Fitness  #
#################################################################

class CEC05RP_F17(CEC05RP_F16):
	def __init__(self, dim_num, bound = [-5, 5]):
		super(CEC05RP_F17, self).__init__(dim_num, bound)

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			fit[i] = self.hybrid_composition(feat)
		fit += 120.
		fit *= (1.0 + 0.2 * np.abs(np.random.randn(population.shape[0])))
		return fit

#################################################################
#                 Hybrid Composition Function 2                 #
#################################################################

class CEC05RP_F18(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F18, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o[0]
		self.min_fitness = 10.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			fit[i] = self.hybrid_composition(feat)
		return fit + 10.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func2_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func2_M_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.array([1., 2., 1.5, 1.5, 1., 1., 1.5, 1.5, 2., 2.])
		self.m_lambda 			= np.array([2.*5./32., 5./32., 2., 1., 10./100., 5./100., 2.*10., 10., 10./60., 5./60.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_o[9] 			= np.zeros(self.dim_num)
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

	@classmethod
	def basic_func(cls, i, x):
		if i <= 1:
			return cls.ackley(x)
		elif i <= 3:
			return cls.rastrigin(x)
		elif i <= 5:
			return cls.sphere(x)
		elif i <= 7:
			return cls.weierstrass(x)
		elif i <= 9:
			return cls.griewank(x)
		else:
			raise NotImplementedError

############################################################################
#  Rotated Hybrid Composition Function 2 with narrow basin global optimum  #
############################################################################

class CEC05RP_F19(CEC05RP_F18):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F19, self).__init__(dim_num, bound)

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func2_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func2_M_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.array([.1, 2., 1.5, 1.5, 1., 1., 1.5, 1.5, 2., 2.])
		self.m_lambda 			= np.array([.1*5./32., 5./32., 2., 1., 10./100., 5./100., 2.*10., 10., 10./60., 5./60.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_o[9] 			= np.zeros(self.dim_num)
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

############################################################################
#  Rotated Hybrid Composition Function 2 with Global Optimum on the Bounds #
############################################################################

class CEC05RP_F20(CEC05RP_F18):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F20, self).__init__(dim_num, bound)

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func2_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func2_M_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.array([1., 2., 1.5, 1.5, 1., 1., 1.5, 1.5, 2., 2.])
		self.m_lambda 			= np.array([2.*5./32., 5./32., 2., 1., 10./100., 5./100., 2.*10., 10., 10./60., 5./60.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_o[9] 			= np.zeros(self.dim_num)
		self.m_o[0] 			= np.ones(self.dim_num) * 5.
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

#################################################################
#                 Hybrid Composition Function 3                 #
#################################################################

class CEC05RP_F21(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F21, self).__init__(dim_num, bound, False)
		self.global_optimum = self.m_o[0]
		self.min_fitness = 360.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			fit[i] = self.hybrid_composition(feat)
		return fit + 360.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func3_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func3_M_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2.])
		self.m_lambda 			= np.array([5.*5./100., 5./100., 5., 1., 5., 1., 5.*10., 10., 25./200., 5./200.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

	@classmethod
	def basic_func(cls, i, x):
		if i <= 1:
			return cls.EScafferF6(x)
		elif i <= 3:
			return cls.rastrigin(x)
		elif i <= 5:
			return cls.F8F2(x)
		elif i <= 7:
			return cls.weierstrass(x)
		elif i <= 9:
			return cls.griewank(x)
		else:
			raise NotImplementedError

###########################################################################
# Rotated Hybrid Composition Function 3 with High Condition Number Matrix #
###########################################################################

class CEC05RP_F22(CEC05RP_F21):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F22, self).__init__(dim_num, bound)

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func3_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func3_HM_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2.])
		self.m_lambda 			= np.array([5.*5./100., 5./100., 5., 1., 5., 1., 5.*10., 10., 25./200., 5./200.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

###########################################################################
#          Non-Continuous Rotated Hybrid Composition Function 3           #
###########################################################################

class CEC05RP_F23(CEC05RP_F21):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F23, self).__init__(dim_num, bound)

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			feat 	= self.myXRound(feat, self.m_o[0])
			fit[i] = self.hybrid_composition(feat)
		return fit + 360.


###########################################################################
#                  Rotated Hybrid Composition Function 4                  #
###########################################################################

class CEC05RP_F24(cec_05_rp_benchmark):
	def __init__(self, dim_num, bound = [-5., 5.]):
		super(CEC05RP_F24, self).__init__(dim_num, bound)
		self.global_optimum = self.m_o[0]
		self.min_fitness = 260.

	def evaluate(self, population):
		population 	= check_shape(population)
		fit 		= np.zeros(population.shape[0])
		for i, feat in enumerate(population):
			fit[i] = self.hybrid_composition(feat)
		return fit + 260.

	def init_params(self):
		DEFAULT_FILE_DATA 		= os.path.join(input_data_path, 'hybrid_func4_data.txt')
		DEFAULT_FILE_MX 		= os.path.join(input_data_path, 'hybrid_func4_M_D%d.txt' % self.dim_num)
		self.num_func 			= 10
		self.m_sigma  			= np.ones(self.num_func) * 2.
		self.m_lambda 			= np.array([10., 5./20., 1., 5./32., 1., 5./100., 5./50., 1., 5./100., 5./100.])
		self.m_func_biases 		= np.arange(0, 1000, 100).astype('float')
		self.C 					= 2000.

		self.m_o 				= self.loadMatrixFromFile(DEFAULT_FILE_DATA, self.num_func, self.dim_num)
		self.m_M 				= self.loadNMatrixFromFile(DEFAULT_FILE_MX, self.num_func, self.dim_num, self.dim_num)
		
		# calculate/estimate the fmax for all the functions involved
		self.find_m_fax()

	@classmethod
	def basic_func(cls, i, x):
		if i == 0:
			return cls.weierstrass(x)
		elif i == 1:
			return cls.EScafferF6(x)
		elif i == 2:
			return cls.F8F2(x)
		elif i == 3:
			return cls.ackley(x)
		elif i == 4:
			return cls.rastrigin(x)
		elif i == 5:
			return cls.griewank(x)
		elif i == 6:
			return cls.EScafferF6NonCont(x)
		elif i == 7:
			return cls.rastriginNonCont(x)
		elif i == 8:
			return cls.elliptic(x)
		elif i == 9:
			return cls.sphere_noise(x)
		else:
			raise NotImplementedError

###########################################################################
#                  Rotated Hybrid Composition Function 4                  #
###########################################################################

class CEC05RP_F25(CEC05RP_F24):
	def __init__(self, dim_num, bound = [2., 5.]):
		super(CEC05RP_F25, self).__init__(dim_num, bound)

	def bounder(self, feature, method):
		return feature
