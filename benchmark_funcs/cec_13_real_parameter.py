import os, sys
import numpy as np 
from copy import copy
from constant import CODE_PATH
from benchmark import Benchmark, check_shape

DIM_NUM 	= 10
BOUND 		= [-100, 100]
###########################################################################################################
#         Benchmark Functions for the CEC'2013 Special Session on Real-Parameter Optimization             #
###########################################################################################################
input_data_path 	= os.path.join(CODE_PATH, 'benchmark_funcs', 'CEC_2013_RP_Data')
input_data_names 	= ['M_D2.txt', 'M_D5.txt', 'M_D10.txt', 'M_D20.txt', 'M_D30.txt', 'M_D40.txt', 'M_D50.txt',\
					   'M_D60.txt', 'M_D70.txt', 'M_D80.txt', 'M_D90.txt', 'M_D100.txt', 'shift_data.txt']


class cec_13_rp_benchmark(Benchmark):
	def __init__(self, dim_num = DIM_NUM, bound = BOUND, maximize = False):
		super(cec_13_rp_benchmark, self).__init__(dim_num, bound, maximize)
		self.init_params()

	def init_params(self):
		cf_num = 10
		if self.dim_num not in [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			raise Exception('the test suite only support 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 dimensional problems ')
		else:
			m_list = []
			s_list = []
			with open(os.path.join(input_data_path, 'M_D%d.txt' % self.dim_num), 'r') as datafile:
				for line in datafile.readlines():
					m_list.extend(map(float, line.strip().split()))
			with open(os.path.join(input_data_path, 'shift_data.txt'), 'r') as datafile:
				for line in datafile.readlines():
					s_list.extend(map(float, line.strip().split()))

			self.mr = np.array(m_list).reshape(cf_num, self.dim_num, self.dim_num)
			#self.os = np.array(s_list).reshape(cf_num, 100)[:, :self.dim_num]
			#self.os = np.array(s_list).reshape(cf_num * 100 / self.dim_num, self.dim_num)
			self.os = np.array(s_list)[: self.dim_num * cf_num].reshape(cf_num, self.dim_num)

########################################################
#			   	    Sphere Function 				   #
########################################################
class CEC13RP_F01(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F01, self).__init__(dim_num = dim_num)
		self.min_fitness = -1400.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 0):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population = rotatefunc(population, self.mr[mr_start])
		fit 		= np.sum(population ** 2, axis = 1)
		return fit - 1400.

########################################################
#      Rotated High Conditioned Elliptic Function      #
########################################################
class CEC13RP_F02(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F02, self).__init__(dim_num = dim_num)
		self.min_fitness = -1300.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= oszfunc(population)
		condition 	= 1E06
		coeffs 		= np.power(condition, np.linspace(0, 1, self.dim_num))
		fit 		= np.sum((population ** 2) * coeffs, axis = 1)
		return fit - 1300.

####################Unimodal Functions##################

########################################################
#             Rotated Bent Cigar Function              #
########################################################
class CEC13RP_F03(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F03, self).__init__(dim_num = dim_num)
		self.min_fitness = -1200.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		beta = 0.5
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= asyfunc(population, beta)
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start + 1])
			#population 	= rotatefunc(population, np.zeros(shape = (10, 10)))
		coeffs 		= np.ones(self.dim_num) * 1E06
		coeffs[0]	= 1.
		fit 		= np.sum((population ** 2) * coeffs, axis = 1)
		return fit - 1200.

########################################################
#               Rotated Discus Function                #
########################################################
class CEC13RP_F04(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F04, self).__init__(dim_num = dim_num)
		self.min_fitness = -1100.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= oszfunc(population)
		coeffs 		= np.ones(self.dim_num)
		coeffs[0] 	= 1E06
		fit 		= np.sum((population ** 2) * coeffs, axis = 1)
		return fit - 1100.

########################################################
#              Different Powers Function               #
########################################################
class CEC13RP_F05(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F05, self).__init__(dim_num = dim_num)
		self.min_fitness = -1000.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 0):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population = rotatefunc(population, self.mr[mr_start])
		coeffs 		= 2. + 4. * np.linspace(0, 1, self.dim_num)
		#coeffs 		= map(int, 2 + 4 * np.linspace(0, 1, self.dim_num))
		fit 		= np.sqrt(np.sum(np.power(np.abs(population), coeffs), axis = 1))
		return fit - 1000.

################Basic Multimodal Functions##############

########################################################
#             Rotated Rosenbrock Function              #
########################################################
class CEC13RP_F06(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F06, self).__init__(dim_num = dim_num)
		self.min_fitness = -900.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start]) * 2.048 / 100
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start]) + 1.
		fit 		= np.sum(100. * (population[:, : -1]**2 - population[:, 1:])**2 + (population[:, : -1] - 1)**2, axis = 1)
		return fit - 900.

########################################################
#             Rotated Schaffer's Function              #
########################################################
''' problematic '''
class CEC13RP_F07(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F07, self).__init__(dim_num = dim_num)
		self.min_fitness = -800.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= asyfunc(population, 0.5)
		#population 	= diagfunc(population, 10)
		#population 	= rotatefunc(population, self.mr[1])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start + 1])
		population 	= diagfunc(population, 10)
		population 	= np.sqrt(population[:, : -1] ** 2 + population[:, 1:] ** 2)
		fit 		= (np.sum((np.sqrt(population) + np.sqrt(population) * (np.sin(50. * np.power(population, 0.2)) ** 2)), axis = 1) / (self.dim_num - 1)) ** 2
		return fit - 800.
		
########################################################
#              Rotated Ackley's Function               #
########################################################
class CEC13RP_F08(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F08, self).__init__(dim_num = dim_num)
		self.min_fitness = -700.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= asyfunc(population, 0.5)
		#population 	= diagfunc(population, 10.)
		#population 	= rotatefunc(population, self.mr[1])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start + 1])
		population 	= diagfunc(population, 10.)
		tmp 		= np.sum((population / np.sqrt(self.dim_num))**2, axis = 1)
		fit 		= 20. - 20. * np.exp(-0.2 * np.sqrt(tmp)) - np.exp(np.sum(np.cos(2 * np.pi * population), axis = 1) / self.dim_num) + np.e
		return fit - 700.

########################################################
#            Rotated Weierstrass Function              #
########################################################
class CEC13RP_F09(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F09, self).__init__(dim_num = dim_num)
		self.min_fitness = -600.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 0.005 * shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= asyfunc(population, 0.5)
		#population 	= diagfunc(population, 10.)
		#population 	= rotatefunc(population, self.mr[1])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start + 1])
			#population 	= rotatefunc(population, np.zeros(shape = (10, 10)))
		population 	= diagfunc(population, 10.)
		tmp 		= map(self.weierstrass, population)
		fit 		= np.sum(tmp, axis = 1)
		return fit - 600.

	@staticmethod
	def weierstrass(x):
		a, b, kmax 	= 0.5, 3, 20
		result = .0
		for k in range(kmax+1):
			result += (a**k) * (np.cos(2 * np.pi * (b**k) * (x + 0.5)) - np.cos(2 * np.pi * (b**k) * 0.5))
		return result

########################################################
#             Rotated Griewank's Function              #
########################################################
class CEC13RP_F10(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F10, self).__init__(dim_num = dim_num)
		self.min_fitness = -500.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 6.0 * shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= diagfunc(population, 100)
		fit 		= 1. / 4000. * np.sum(population * population, axis = 1) -\
					  np.multiply.reduce(np.cos(population / np.sqrt((np.arange(self.dim_num) + 1.))), axis = 1) + 1.
		return fit - 500.

########################################################
#                Rastrigin's Function                  #
########################################################
class CEC13RP_F11(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F11, self).__init__(dim_num = dim_num)
		self.min_fitness = -400.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 0.0512 * shiftfunc(population, self.os[os_start])
		population 	= oszfunc(population)
		population 	= asyfunc(population, 0.2)
		population 	= diagfunc(population, 10)
		fit 		= 10. * self.dim_num + np.sum(population**2 - 10.*np.cos(2*np.pi*population), axis = 1)
		return fit - 400.

########################################################
#             Rotated Rastrigin's Function             #
########################################################
class CEC13RP_F12(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F12, self).__init__(dim_num = dim_num)
		self.min_fitness = -300.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 0.0512 * shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= oszfunc(population)
		population 	= asyfunc(population, 0.2)
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start + 1])
		population 	= diagfunc(population, 10)
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		fit 		= 10. * self.dim_num + np.sum(population**2 - 10.*np.cos(2*np.pi*population), axis = 1)
		return fit - 300.

########################################################
#     Non-Continuous Rotated Rastrigin's Function      #
########################################################
class CEC13RP_F13(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F13, self).__init__(dim_num = dim_num)
		self.min_fitness = -200.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 0.0512 * shiftfunc(population, self.os[0])
		if r_flag:
			population 	= rotatefunc(population, self.mr[0])
		tmp 		= np.abs(population)
		mask_1 		= tmp <= 0.5
		mask_2 		= tmp > 0.5
		population 	= population * mask_1 + np.round(2 * population) / 2 * mask_2
		population 	= oszfunc(population)
		population 	= asyfunc(population, 0.2)
		if r_flag:
			population 	= rotatefunc(population, self.mr[1])
		population 	= diagfunc(population, 10)
		population 	= rotatefunc(population, self.mr[0])
		fit 		= 10. * self.dim_num + np.sum(population**2 - 10.*np.cos(2*np.pi*population), axis = 1)
		return fit - 200.


########################################################
#                 Schwefel's Function                  #
########################################################
class CEC13RP_F14(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F14, self).__init__(dim_num = dim_num)
		self.min_fitness = -100.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 0):
		population 	= check_shape(population)
		population 	= 10. * shiftfunc(population, self.os[os_start])
		population 	= diagfunc(population, 10) + 4.209687462275036e+002
		mask_2 		= population > 500.
		mask_3 		= population < -500.
		mask_1 		= 1 - mask_2 - mask_3
		population 	= mask_1 * (population * np.sin(np.sqrt(np.abs(population)))) +\
					  mask_2 * ((500. - np.mod(population, 500)) * np.sin(np.sqrt(np.abs(500. - np.mod(population, 500)))) - (population - 500.) ** 2 / 10000 / self.dim_num) +\
					  mask_3 * ((np.mod(np.abs(population), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(population), 500) - 500))) - (population + 500.) ** 2 / 10000 / self.dim_num)
		fit 		= 4.189828872724338e+002 * self.dim_num - np.sum(population, axis = 1)
		return fit - 100.

########################################################
#             Rotated Schwefel's Function              #
########################################################
class CEC13RP_F15(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F15, self).__init__(dim_num = dim_num)
		self.min_fitness = 100.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 10. * shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= diagfunc(population, 10) + 4.209687462275036e+002
		mask_2 		= population > 500.
		mask_3 		= population < -500.
		mask_1 		= 1 - mask_2 - mask_3
		population 	= mask_1 * (population * np.sin(np.sqrt(np.abs(population)))) +\
					  mask_2 * ((500. - np.mod(population, 500)) * np.sin(np.sqrt(np.abs(500. - np.mod(population, 500)))) - (population - 500.) ** 2 / 10000 / self.dim_num) +\
					  mask_3 * ((np.mod(np.abs(population), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(population), 500) - 500))) - (population + 500.) ** 2 / 10000 / self.dim_num)
		fit 		= 4.189828872724338e+002 * self.dim_num - np.sum(population, axis = 1)
		return fit + 100.


########################################################
#               Rotated Katsuura Function              #
########################################################
class CEC13RP_F16(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F16, self).__init__(dim_num = dim_num)
		self.min_fitness = 200.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		pop_num,dim_num = population.shape
		population 	= check_shape(population)
		population 	= 0.05 * shiftfunc(population, self.os[0])
		if r_flag:
			population 	= rotatefunc(population, self.mr[0])
		population 	= diagfunc(population, 100)
		if r_flag:
			population 	= rotatefunc(population, self.mr[1])
		#tmp 		= map(self.inner, population)
		tmp 		= np.zeros(shape = (pop_num, dim_num))
		for i in range(pop_num):
			for j in range(dim_num):
				tmp[i, j] = self.inner(population[i, j])
		fit 		= 10. / (self.dim_num**2) *np.product(np.power(1 + (np.arange(self.dim_num).astype('float') + 1.) * tmp, 10. / np.power(self.dim_num, 1.2)), axis = 1) -\
					  10. / (self.dim_num**2)
		return fit + 200.

	@staticmethod
	def inner(z):
		j = np.arange(1, 33).astype('float')
		p = np.power(2, j)
		tmp = np.abs(p * z - np.round(p * z)) / p
		return np.sum(tmp)

########################################################
#             Lunacek Bi_Rastrigin Function            #
########################################################
class CEC13RP_F17(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F17, self).__init__(dim_num = dim_num)
		self.s  = 1. - 1. / (2. * np.sqrt(self.dim_num + 20) - 8.2)
		self.u0 = 2.5
		self.u1 = -np.sqrt((self.u0**2 - 1) / self.s)
		self.min_fitness = 300.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 0):
		population 	= check_shape(population)
		y 			= 0.1 * shiftfunc(population, self.os[0])
		x_hat 		= 2. * np.sign(self.os[0]) * y + self.u0
		z 			= diagfunc(x_hat - self.u0, 100)
		tmp1 		= np.sum((x_hat - self.u0)**2, axis = 1)
		tmp2 		= self.dim_num + self.s * np.sum((x_hat - self.u1)**2, axis = 1)
		fit 		= np.min(np.vstack((tmp1, tmp2)), axis = 0) + 10. * (self.dim_num - np.sum(np.cos(2 * np.pi * z) , axis = 1))
		return fit + 300.


########################################################
#         Rotated Lunacek Bi_Rastrigin Function        #
########################################################
class CEC13RP_F18(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F18, self).__init__(dim_num = dim_num)
		self.s  = 1. - 1. / (2. * np.sqrt(self.dim_num + 20) - 8.2)
		self.u0 = 2.5
		self.u1 = -np.sqrt((self.u0**2 - 1) / self.s)
		self.min_fitness = 400.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		y 			= 0.1 * shiftfunc(population, self.os[0])
		x_hat 		= 2. * np.sign(self.os[0]) * y + self.u0
		z 			= rotatefunc(diagfunc(rotatefunc(x_hat - self.u0, self.mr[0]), 100), self.mr[1])
		tmp1 		= np.sum((x_hat - self.u0)**2, axis = 1)
		tmp2 		= self.dim_num + self.s * np.sum((x_hat - self.u1)**2, axis = 1)
		fit 		= np.min(np.vstack((tmp1, tmp2)), axis = 0) + 10. * (self.dim_num - np.sum(np.cos(2 * np.pi * z) , axis = 1))
		return fit + 400.

########################################################
#    Expanded Griewank's plus Rosenbrock's Function    #
########################################################
class CEC13RP_F19(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F19, self).__init__(dim_num = dim_num)
		self.min_fitness = 500.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= 0.05 * shiftfunc(population, self.os[0])
		if r_flag:
			population 	= rotatefunc(population, self.mr[0]) + 1.
		fit 	= 0.
		for i, j in zip(np.arange(self.dim_num), np.roll(np.arange(self.dim_num), -1)):
			fit += self.g1(self.g2(population[:, i], population[:, j]))
		return fit + 500.
	@staticmethod
	def g1(x):
		return (x**2) / 4000. - np.cos(x) + 1.
	@staticmethod
	def g2(x1, x2):
		return 100. * ((x1**2 - x2)**2) + (x1 - 1)**2

########################################################
#       Rotated Expanded Schaffer's F6 Function        #
########################################################
class CEC13RP_F20(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F20, self).__init__(dim_num = dim_num)
		self.min_fitness = 600.

	def evaluate(self, population, os_start = 0, mr_start = 0, r_flag = 1):
		population 	= check_shape(population)
		population 	= shiftfunc(population, self.os[os_start])
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start])
		population 	= asyfunc(population, 0.5)
		if r_flag:
			population 	= rotatefunc(population, self.mr[mr_start + 1])
			#population 		= rotatefunc(population, np.zeros(shape = (self.dim_num, self.dim_num)))
		fit 		= 0.
		for i, j in zip(np.arange(self.dim_num), np.roll(np.arange(self.dim_num), -1)):
			fit += self.schaffer(population[:, i], population[:, j])
		return fit + 600.
	@staticmethod
	def schaffer(x1, x2):
		return 0.5 + (np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5) / ((1. + 0.001 * (x1**2 + x2**2))**2)


########################################################
#         Composition Function 1 (n=5, Rotated)        #
########################################################
class CEC13RP_F21(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F21, self).__init__(dim_num = dim_num)
		self.min_fitness = 700.

	def evaluate(self, population, r_flag = 1):
		pop_num	= population.shape[0]
		cf_num 	= 5
		sigm 	= np.arange(10, 60, 10).astype('float')
		labd 	= np.array([1., 1E-06, 1E-26, 1E-6, 0.1])
		bias 	= np.arange(0, 500, 100).astype('float')
		fits 	= np.zeros(shape = (cf_num, pop_num))
		
		b1 		= f6(self.dim_num)
		fits[0] = (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) + 900.) * labd[0]
		b2 		= f5(self.dim_num)
		fits[1] = (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 1000.) * labd[1]
		b3 		= f3(self.dim_num)
		fits[2] = (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) + 1200.) * labd[2]
		b4 		= f4(self.dim_num)
		fits[3] = (b4.evaluate(population, os_start = 3, mr_start = 3, r_flag = r_flag) + 1100.) * labd[3]
		b5 		= f1(self.dim_num)
		fits[4] = (b5.evaluate(population, os_start = 4, mr_start = 4, r_flag = r_flag) + 1400.) * labd[4]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 700.

########################################################
#        Composition Function 2 (n=3, Unrotated)       #
########################################################
class CEC13RP_F22(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F22, self).__init__(dim_num = dim_num)
		self.min_fitness = 800.

	def evaluate(self, population, r_flag = 0):
		pop_num = population.shape[0]
		cf_num 	= 3
		sigm 	= np.ones(cf_num) * 20.
		labd 	= np.ones(cf_num)
		bias 	= np.array([0., 100., 200.])
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f14(self.dim_num)
		fits[0] = (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) + 100.) * labd[0]
		b2 		= f14(self.dim_num)
		fits[1]	= (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 100.) * labd[1]
		b3 		= f14(self.dim_num)
		fits[2] = (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) + 100.) * labd[2]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 800.


########################################################
#        Composition Function 3 (n=3, Rotated)         #
########################################################
class CEC13RP_F23(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F23, self).__init__(dim_num = dim_num)
		self.min_fitness = 900.

	def evaluate(self, population, r_flag = 1):
		pop_num = population.shape[0]
		cf_num 	= 3
		sigm 	= np.ones(cf_num) * 20.
		labd 	= np.ones(cf_num)
		bias 	= np.array([0., 100., 200.])
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f15(self.dim_num)
		fits[0] = (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) - 100.) * labd[0]
		b2 		= f15(self.dim_num)
		fits[1]	= (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) - 100.) * labd[1]
		b3 		= f15(self.dim_num)
		fits[2] = (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) - 100.) * labd[2]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 900.


########################################################
#        Composition Function 4 (n=3, Rotated)         #
########################################################
class CEC13RP_F24(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F24, self).__init__(dim_num = dim_num)
		self.min_fitness = 1000.

	def evaluate(self, population, r_flag = 1):
		pop_num = population.shape[0]
		cf_num 	= 3
		sigm 	= np.ones(cf_num) * 20.
		labd 	= np.array([0.25, 1., 2.5])
		bias 	= np.array([0., 100., 200.])
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f15(self.dim_num)
		fits[0]	= (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) - 100.) * labd[0]
		b2 		= f12(self.dim_num)
		fits[1]	= (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 300.) * labd[1]
		b3 		= f9(self.dim_num)
		fits[2]	= (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) + 600.) * labd[2]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 1000.

########################################################
#        Composition Function 5 (n=3, Rotated)         #
########################################################
class CEC13RP_F25(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F25, self).__init__(dim_num = dim_num)
		self.min_fitness = 1100.

	def evaluate(self, population, r_flag = 1):
		pop_num = population.shape[0]
		cf_num 	= 3
		sigm 	= np.array([10, 30, 50])
		labd 	= np.array([0.25, 1., 2.5])
		bias 	= np.array([0., 100., 200.])
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f15(self.dim_num)
		fits[0]	= (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) - 100.) * labd[0]
		b2 		= f12(self.dim_num)
		fits[1]	= (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 300.) * labd[1]
		b3 		= f9(self.dim_num)
		fits[2]	= (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) + 600.) * labd[2]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 1100.


########################################################
#        Composition Function 6 (n=5, Rotated)         #
########################################################
class CEC13RP_F26(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F26, self).__init__(dim_num = dim_num)
		self.min_fitness = 1200.

	def evaluate(self, population, r_flag = 1):
		pop_num	= population.shape[0]
		cf_num 	= 5
		sigm 	= np.ones(cf_num) * 10.
		labd 	= np.array([0.25, 1., 1E-7, 2.5, 10.])
		bias 	= np.arange(0, 500, 100).astype('float')
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f15(self.dim_num)
		fits[0]	= (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) - 100.) * labd[0]
		b2 		= f12(self.dim_num)
		fits[1] = (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 300.) * labd[1]
		b3 		= f2(self.dim_num)
		fits[2] = (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) + 1300.) * labd[2]
		b4 		= f9(self.dim_num)
		fits[3] = (b4.evaluate(population, os_start = 3, mr_start = 3, r_flag = r_flag) + 600.) * labd[3]
		b5 		= f10(self.dim_num)
		fits[4] = (b5.evaluate(population, os_start = 4, mr_start = 4, r_flag = r_flag) + 500.) * labd[4]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 1200.

########################################################
#        Composition Function 7 (n=5, Rotated)         #
########################################################
class CEC13RP_F27(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F27, self).__init__(dim_num = dim_num)
		self.min_fitness = 1300.

	def evaluate(self, population, r_flag = 1):
		pop_num	= population.shape[0]
		cf_num 	= 5
		sigm 	= np.array([10, 10, 10, 20, 20])
		labd 	= np.array([100., 10., 2.5, 25., 0.1])
		bias 	= np.arange(0, 500, 100).astype('float')
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f10(self.dim_num)
		fits[0] = (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) + 500.) * labd[0]
		b2 		= f12(self.dim_num)
		fits[1] = (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 300.) * labd[1]
		b3 		= f15(self.dim_num)
		fits[2] = (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) - 100.) * labd[2]
		b4 		= f9(self.dim_num)
		fits[3] = (b4.evaluate(population, os_start = 3, mr_start = 3, r_flag = r_flag) + 600.) * labd[3]
		b5 		= f1(self.dim_num)
		fits[4]	= (b5.evaluate(population, os_start = 4, mr_start = 4, r_flag = r_flag) + 1400.) * labd[4]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 1300.

########################################################
#        Composition Function 8 (n=5, Rotated)         #
########################################################
class CEC13RP_F28(cec_13_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC13RP_F28, self).__init__(dim_num = dim_num)
		self.min_fitness = 1400.

	def evaluate(self, population, r_flag = 1):
		pop_num	= population.shape[0]
		cf_num 	= 5
		sigm 	= np.arange(10, 60, 10).astype('float')
		labd 	= np.array([2.5, 2.5E-3, 2.5, 5E-4, 0.1])
		bias 	= np.arange(0, 500, 100).astype('float')
		fits 	= np.zeros(shape = (cf_num, pop_num))

		b1 		= f19(self.dim_num)
		fits[0] = (b1.evaluate(population, os_start = 0, mr_start = 0, r_flag = r_flag) - 500.) * labd[0]
		b2 		= f7(self.dim_num)
		fits[1] = (b2.evaluate(population, os_start = 1, mr_start = 1, r_flag = r_flag) + 800.) * labd[1]
		b3 		= f15(self.dim_num)
		fits[2] = (b3.evaluate(population, os_start = 2, mr_start = 2, r_flag = r_flag) - 100.) * labd[2]
		b4 		= f20(self.dim_num)
		fits[3] = (b4.evaluate(population, os_start = 3, mr_start = 3, r_flag = r_flag) - 600.) * labd[3]
		b5 		= f1(self.dim_num)
		fits[4]	= (b5.evaluate(population, os_start = 4, mr_start = 4, r_flag = r_flag) + 1400.) * labd[4]

		final_fits = np.zeros(pop_num)
		for i in range(pop_num):
			final_fits[i] = cf_cal(population[i], self.dim_num, cf_num, fits[:, i], sigm, bias, self.os)
		return final_fits + 1400.


########################################################
#				   Helpler Functions				   #
########################################################

def shiftfunc(population, os):
	return population - os

def rotatefunc(population, mr):
	return np.dot(population, mr.transpose())

def asyfunc(population, beta):
	P, D = population.shape
	new_population = copy(population)
	temp = np.repeat(beta * np.linspace(0, 1, D).reshape(1, D), P, axis = 0)
	pos = population > 0
	new_population[pos] = np.power(population[pos], 1. + temp[pos] * np.sqrt(population[pos]))
	return new_population

def oszfunc(population):
	new_population = copy(population)
	tmp 	= new_population[:, [0, -1]]
	pos = tmp > 0
	neg = tmp < 0
	tmp[pos] 	= np.log(tmp[pos])
	tmp[pos] 	= np.exp(tmp[pos] + 0.049 * (np.sin(10. * tmp[pos]) + np.sin(7.9 * tmp[pos])))
	tmp[neg] 	= np.log(-tmp[neg])
	tmp[neg]	= -np.exp(tmp[neg] + 0.049 * (np.sin(5.5 * tmp[neg]) + np.sin(3.1 * tmp[neg])))
	new_population[:, [0, -1]] = tmp
	return new_population

def diagfunc(population, alpha):
	pop_num, dim_num = population.shape
	new_population = np.power(np.sqrt(alpha), np.linspace(0, 1, dim_num)) * population
	return new_population

def cf_cal(x, dim_num, cf_num, fits, sigmas, bias, os):
	w = np.zeros(cf_num)
	for i in range(cf_num):
		fits[i] += bias[i]
		tmp = np.sum(np.abs(x - os[i]))
		if tmp == 0.:
			w[i] = float('inf')
		else:
			w[i] = 1. / np.sqrt(np.sum((x - os[i])**2)) * np.exp(- np.sum((x - os[i])**2) / (2. * dim_num * (sigmas[i] ** 2)))
	if np.sum(np.isinf(w)) != 0:
		inf_mask = np.isinf(w)
		fin_mask = np.isfinite(w)
		w[inf_mask] = 1.
		w[fin_mask] = 0.
	if np.sum(np.abs(w)) == 0.:
		w = np.ones(cf_num)
	w = w / np.sum(w)
	return np.sum(w * fits)


