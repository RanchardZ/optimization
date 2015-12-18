import os, sys
import numpy as np 
from copy import copy
from math import ceil, floor
from constant import CODE_PATH
from benchmark import Benchmark, check_shape


###########################################################################################################
#         Benchmark Functions for the CEC'2014 Special Session on Real-Parameter Optimization             #
###########################################################################################################
input_data_path 	= os.path.join(CODE_PATH, 'benchmark_funcs', 'CEC_2014_RP_Data')

class cec_14_rp_benchmark(Benchmark):
	def __init__(self, func_num, dim_num, bound = [-100., 100.], maximize = False):
		super(cec_14_rp_benchmark, self).__init__(dim_num, bound, maximize)
		self.func_num = func_num
		self.init_params()

	def init_params(self):
		cf_num = 10
		if self.dim_num not in [2, 10, 20, 30, 50, 100]:
			raise Exception('Error: Test functions are only defined for D=2,10,20,30,50,100.')
		if self.dim_num == 2 and ((self.func_num>=17 and self.func_num<=22) or (self.func_num>=29 and self.func_num<=30)):
			raise Exception('Error: hf01,hf02,hf03,hf04,hf05,hf06,cf07&cf08 are NOT defined for D=2.')
		# load matrix M
		with open(os.path.join(input_data_path, "M_%d_D%d.txt" % (self.func_num, self.dim_num))) as m_data:
			if self.func_num < 23:
				self.M = np.zeros(shape=(self.dim_num, self.dim_num))
				for i, row_data in enumerate(m_data.readlines()):
					self.M[i] = map(float, row_data.strip().split())
			else:
				self.M = np.zeros(shape=(cf_num, self.dim_num, self.dim_num))
				count = -1
				for i, row_data in enumerate(m_data.readlines()):
					r = i % self.dim_num
					if r == 0:
						count += 1
					self.M[count, r] = map(float, row_data.strip().split())
		
		# load shift data
		if self.func_num < 23:
			self.OShift = np.zeros(self.dim_num)
			with open(os.path.join(input_data_path, "shift_data_%d.txt" % self.func_num)) as shift_data:
				self.OShift = map(float, shift_data.readline().strip().split())[: self.dim_num]
		else:
			self.OShift = np.zeros(shape=(cf_num, self.dim_num))
			with open(os.path.join(input_data_path, "shift_data_%d.txt" % self.func_num)) as shift_data:
				for i, row_data in enumerate(shift_data.readlines()):
					self.OShift[i] = map(float, row_data.strip().split())[: self.dim_num]

		# load shuffle data
		if (self.func_num>=17) and (self.func_num<=22):
			with open(os.path.join(input_data_path, "shuffle_data_%d_D%d.txt" % (self.func_num, self.dim_num))) as shuffle_data:
				self.SS = np.array(map(int, shuffle_data.readline().strip().split()))
		elif (self.func_num==29) or (self.func_num==30):
			self.SS = np.zeros(shape=(cf_num, self.dim_num)).astype('int')
			SS = np.zeros(cf_num*self.dim_num)
			with open(os.path.join(input_data_path, "shuffle_data_%d_D%d.txt" % (self.func_num, self.dim_num))) as shuffle_data:
				SS = map(int, shuffle_data.readline().strip().split())
			for i in range(cf_num):
				self.SS[i] = SS[i*self.dim_num: (i+1)*self.dim_num]

	@staticmethod
	def ellips_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		condition 	= 1E06
		coeffs 		= np.power(condition, np.linspace(0, 1, dim_num))
		fit 		= np.sum((population ** 2) * coeffs, axis = 1)
		return fit
	@staticmethod
	def bent_cigar_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		coeffs 		= np.ones(dim_num) * 1E06
		coeffs[0]	= 1.
		fit 		= np.sum((population ** 2) * coeffs, axis = 1)
		return fit
	@staticmethod
	def discus_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		coeffs 		= np.ones(dim_num)
		coeffs[0] 	= 1E06
		fit 		= np.sum((population ** 2) * coeffs, axis = 1)
		return fit
	@staticmethod
	def rosenbrock_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 2.048/100., s_flag, r_flag) + 1.
		fit 		= np.sum(100. * (population[:, : -1]**2 - population[:, 1:])**2 + (population[:, : -1] - 1)**2, axis = 1)
		return fit
	@staticmethod
	def ackley_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		tmp 		= np.sum((population / np.sqrt(dim_num))**2, axis = 1)
		fit 		= 20. - 20. * np.exp(-0.2 * np.sqrt(tmp)) - np.exp(np.sum(np.cos(2 * np.pi * population), axis = 1) / dim_num) + np.e
		return fit
	@classmethod
	def weierstrass_func(self, population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 0.5/100., s_flag, r_flag)
		tmp 		= map(self.weierstrass, population)
		fit 		= np.sum(tmp, axis = 1)
		return fit
	@staticmethod
	def weierstrass(x):
		a, b, kmax 	= 0.5, 3, 20
		result = .0
		for k in range(kmax+1):
			result += (a**k) * (np.cos(2 * np.pi * (b**k) * (x + 0.5)) - np.cos(2 * np.pi * (b**k) * 0.5))
		return result
	@staticmethod
	def griewank_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 600./100., s_flag, r_flag)
		fit 		= 1. / 4000. * np.sum(population * population, axis = 1) -\
					  np.multiply.reduce(np.cos(population / np.sqrt((np.arange(dim_num) + 1.))), axis = 1) + 1.
		return fit
	@staticmethod
	def rastrigin_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 5.12/100., s_flag, r_flag)
		fit 		= 10. * dim_num + np.sum(population**2 - 10.*np.cos(2*np.pi*population), axis = 1)
		return fit
	@staticmethod
	def schwefel_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1000./100., s_flag, r_flag)
		population 	+= 4.209687462275036e+002
		mask_2 		= population > 500.
		mask_3 		= population < -500.
		mask_1 		= 1 - mask_2 - mask_3
		population 	= mask_1 * (population * np.sin(np.sqrt(np.abs(population)))) +\
					  mask_2 * ((500. - np.mod(population, 500)) * np.sin(np.sqrt(np.abs(500. - np.mod(population, 500)))) - (population - 500.) ** 2 / 10000 / dim_num) +\
					  mask_3 * ((np.mod(np.abs(population), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(population), 500) - 500))) - (population + 500.) ** 2 / 10000 / dim_num)
		fit 		= 4.189828872724338e+002 * dim_num - np.sum(population, axis = 1)
		return fit
	@classmethod
	def katsuura_func(self, population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		pop_num  	= population.shape[0]
		population 	= srfunc(population, dim_num, Os, Mr, 5./100., s_flag, r_flag)
		tmp 		= np.zeros(shape=(pop_num, dim_num))
		for i in range(pop_num):
			for j in range(dim_num):
				tmp[i, j] = self.inner(population[i, j])
		fit 		= 10. / (dim_num**2) *np.product(np.power(1 + (np.arange(dim_num).astype('float') + 1.) * tmp, 10. / np.power(dim_num, 1.2)), axis = 1) -\
					  10. / (dim_num**2)
		return fit
	@staticmethod
	def inner(z):
		j = np.arange(1, 33).astype('float')
		p = np.power(2, j)
		tmp = np.abs(p * z - np.round(p * z)) / p
		return np.sum(tmp)

	@staticmethod
	def happycat_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 5./100., s_flag, r_flag) - 1.
		sum1 		= np.sum(population**2, axis = 1)
		sum2 		= np.sum(population, axis = 1)
		tmp1 		= np.power(np.abs(sum1 - dim_num), 0.25)
		tmp2 		= (0.5*sum1 + sum2) / dim_num + 0.5
		fit 		= tmp1 + tmp2
		return fit
	@staticmethod
	def hgbat_func(population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 5./100., s_flag, r_flag) - 1.
		sum1 		= np.sum(population**2, axis = 1)
		sum2 		= np.sum(population, axis = 1)
		tmp1 		= np.power(np.abs(sum1**2 - sum2**2), 0.5)
		tmp2 		= (0.5*sum1 + sum2) / dim_num + 0.5
		fit 		= tmp1 + tmp2
		return fit
	@classmethod
	def grie_rosen_func(self, population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 5./100., s_flag, r_flag) + 1.
		fit 	= 0.
		for i, j in zip(np.arange(dim_num), np.roll(np.arange(dim_num), -1)):
			fit += self.g1(self.g2(population[:, i], population[:, j]))
		return fit
	@staticmethod
	def g1(x):
		return (x**2) / 4000. - np.cos(x) + 1.
	@staticmethod
	def g2(x1, x2):
		return 100. * ((x1**2 - x2)**2) + (x1 - 1)**2
	@classmethod
	def escaffer6_func(self, population, dim_num, Os, Mr, s_flag, r_flag):
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		fit 		= 0.
		for i, j in zip(np.arange(dim_num), np.roll(np.arange(dim_num), -1)):
			fit += self.schaffer(population[:, i], population[:, j])
		return fit
	@staticmethod
	def schaffer(x1, x2):
		return 0.5 + (np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5) / ((1. + 0.001 * (x1**2 + x2**2))**2)
	@classmethod
	def hf01(self, population, dim_num, Os, Mr, S, s_flag, r_flag):
		cf_num 	= 3
		Gp 		= [0.3, 0.3, 0.4]
		G 		= [0] * cf_num
		G_nx 	= [0] * cf_num
		tmp = 0
		for i in range(cf_num-1):
			G_nx[i] = int(ceil(Gp[i] * dim_num))
			tmp += G_nx[i]
		G_nx[cf_num-1] = dim_num - tmp
		G[0] = 0
		for i in range(1, cf_num):
			G[i] = G[i-1] + G_nx[i-1]
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		population  = population[:, S-1]

		fit = np.zeros(shape = (cf_num, population.shape[0]))

		# schwefel_func
		nx = dim_num
		i = 0
		fit[i] = self.schwefel_func(population[:, :G_nx[i]], G_nx[i], Os[: G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 1
		fit[i] = self.rastrigin_func(population[:, G_nx[i-1]: G_nx[i-1]+G_nx[i]], G_nx[i], Os[G_nx[i-1]: G_nx[i-1]+G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 2
		fit[i] = self.ellips_func(population[:, G_nx[i-2]+G_nx[i-1]:], G_nx[i], Os[G_nx[i-2]+G_nx[i-1]:], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		return np.sum(fit, axis = 0)
	@classmethod
	def hf02(self, population, dim_num, Os, Mr, S, s_flag, r_flag):
		cf_num 	= 3
		Gp 		= [0.3, 0.3, 0.4]
		G 		= [0] * cf_num
		G_nx 	= [0] * cf_num 
		tmp = 0
		for i in range(cf_num-1):
			G_nx[i] = int(ceil(Gp[i] * dim_num))
			tmp += G_nx[i]
		G_nx[cf_num-1] = dim_num - tmp
		G[0] = 0
		for i in range(1, cf_num):
			G[i] = G[i-1] + G_nx[i-1]
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		population  = population[:, S-1]

		fit = np.zeros(shape = (cf_num, population.shape[0]))

		# schwefel_func
		nx = dim_num
		i = 0
		fit[i] = self.bent_cigar_func(population[:, :G_nx[i]], G_nx[i], Os[: G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 1
		fit[i] = self.hgbat_func(population[:, G_nx[i-1]: G_nx[i-1]+G_nx[i]], G_nx[i], Os[G_nx[i-1]: G_nx[i-1]+G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 2
		fit[i] = self.rastrigin_func(population[:, G_nx[i-2]+G_nx[i-1]:], G_nx[i], Os[G_nx[i-2]+G_nx[i-1]:], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		return np.sum(fit, axis = 0)
	@classmethod
	def hf03(self, population, dim_num, Os, Mr, S, s_flag, r_flag):
		cf_num 	= 4
		Gp 		= [0.2, 0.2, 0.3, 0.3]
		G 		= [0] * cf_num
		G_nx 	= [0] * cf_num 
		tmp = 0
		for i in range(cf_num-1):
			G_nx[i] = int(ceil(Gp[i] * dim_num))
			tmp += G_nx[i]
		G_nx[cf_num-1] = dim_num - tmp
		G[0] = 0
		for i in range(1, cf_num):
			G[i] = G[i-1] + G_nx[i-1]
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		population  = population[:, S-1]

		fit = np.zeros(shape = (cf_num, population.shape[0]))

		# schwefel_func
		nx = dim_num
		i = 0
		fit[i] = self.griewank_func(population[:, :G_nx[i]], G_nx[i], Os[: G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 1
		fit[i] = self.weierstrass_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 2
		fit[i] = self.rosenbrock_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 3
		fit[i] = self.escaffer6_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		return np.sum(fit, axis = 0)
	@classmethod
	def hf04(self, population, dim_num, Os, Mr, S, s_flag, r_flag):
		cf_num 	= 4
		Gp 		= [0.2, 0.2, 0.3, 0.3]
		G 		= [0] * cf_num
		G_nx 	= [0] * cf_num 
		tmp = 0
		for i in range(cf_num-1):
			G_nx[i] = int(ceil(Gp[i] * dim_num))
			tmp += G_nx[i]
		G_nx[cf_num-1] = dim_num - tmp
		G[0] = 0
		for i in range(1, cf_num):
			G[i] = G[i-1] + G_nx[i-1]
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		population  = population[:, S-1]

		fit = np.zeros(shape = (cf_num, population.shape[0]))

		# schwefel_func
		nx = dim_num
		i = 0
		fit[i] = self.hgbat_func(population[:, :G_nx[i]], G_nx[i], Os[: G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 1
		fit[i] = self.discus_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 2
		fit[i] = self.grie_rosen_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 3
		fit[i] = self.rastrigin_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		return np.sum(fit, axis = 0)
	@classmethod
	def hf05(self, population, dim_num, Os, Mr, S, s_flag, r_flag):
		cf_num 	= 5
		Gp 		= [0.1,0.2,0.2,0.2,0.3]
		G 		= [0] * cf_num
		G_nx 	= [0] * cf_num 
		tmp = 0
		for i in range(cf_num-1):
			G_nx[i] = int(ceil(Gp[i] * dim_num))
			tmp += G_nx[i]
		G_nx[cf_num-1] = dim_num - tmp
		G[0] = 0
		for i in range(1, cf_num):
			G[i] = G[i-1] + G_nx[i-1]
		population 	= check_shape(population)
		population 	= srfunc(population, dim_num, Os, Mr, 1., s_flag, r_flag)
		population  = population[:, S-1]

		fit = np.zeros(shape = (cf_num, population.shape[0]))

		# schwefel_func
		nx = dim_num
		i = 0
		fit[i] = self.escaffer6_func(population[:, :G_nx[i]], G_nx[i], Os[: G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 1
		fit[i] = self.hgbat_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 2
		fit[i] = self.rosenbrock_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 3
		fit[i] = self.schwefel_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 4
		fit[i] = self.ellips_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		return np.sum(fit, axis = 0)
	@classmethod
	def hf06(self, population, nx, Os, Mr, S, s_flag, r_flag):
		cf_num 	= 5
		Gp 		= [0.1,0.2,0.2,0.2,0.3]
		G 		= [0] * cf_num
		G_nx 	= [0] * cf_num 
		tmp = 0
		for i in range(cf_num-1):
			G_nx[i] = int(ceil(Gp[i] * nx))
			tmp += G_nx[i]
		G_nx[cf_num-1] = nx - tmp
		G[0] = 0
		for i in range(1, cf_num):
			G[i] = G[i-1] + G_nx[i-1]
		population 	= check_shape(population)
		population 	= srfunc(population, nx, Os, Mr, 1., s_flag, r_flag)
		population  = population[:, S-1]

		fit = np.zeros(shape = (cf_num, population.shape[0]))


		i = 0
		fit[i] = self.katsuura_func(population[:, :G_nx[i]], G_nx[i], Os[: G_nx[i]], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 1
		fit[i] = self.happycat_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 2
		fit[i] = self.grie_rosen_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 3
		fit[i] = self.schwefel_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		i = 4
		fit[i] = self.ackley_func(population[:, sum(G_nx[:i]): sum(G_nx[:i+1])], G_nx[i], Os[sum(G_nx[:i]): sum(G_nx[:i+1])], Mr[i*nx: i*nx + G_nx[i]], 0, 0)
		return np.sum(fit, axis = 0)
	@classmethod
	def cf01(self, population, nx, Os, Mr, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 5
		delta 	= np.array([10, 20, 30, 40, 50])
		bias 	= np.arange(0, 500, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.rosenbrock_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e4

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.ellips_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e10

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.bent_cigar_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e30

		i = 3
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.discus_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e10

		i = 4
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.ellips_func(population, nx, tOs, tMr, 1, 0)
		fit[i] = 10000 * fit[i] / 1e10

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf02(self, population, nx, Os, Mr, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 3
		delta 	= np.array([20, 20, 20])
		bias 	= np.arange(0, 300, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.schwefel_func(population, nx, tOs, tMr, 1, 0)

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.rastrigin_func(population, nx, tOs, tMr, 1, r_flag)

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.hgbat_func(population, nx, tOs, tMr, 1, r_flag)

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf03(self, population, nx, Os, Mr, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 3
		delta 	= np.array([10, 30, 50])
		bias 	= np.arange(0, 300, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.schwefel_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 4e3

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.rastrigin_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 1e3

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.ellips_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 1e10

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf04(self, population, nx, Os, Mr, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 5
		delta 	= np.array([10, 10, 10, 10, 10])
		bias 	= np.arange(0, 500, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.schwefel_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 4e3

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.happycat_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 1e3

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.ellips_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 1e10

		i = 3
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.weierstrass_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 400

		i = 4
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.griewank_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 1000 * fit[i] / 100

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf05(self, population, nx, Os, Mr, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 5
		delta 	= np.array([10, 10, 10, 20, 20])
		bias 	= np.arange(0, 500, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.hgbat_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e3

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.rastrigin_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e3

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.schwefel_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 4e3

		i = 3
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.weierstrass_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 400

		i = 4
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.ellips_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e10

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf06(self, population, nx, Os, Mr, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 5
		delta 	= np.array([10, 20, 30, 40, 50])
		bias 	= np.arange(0, 500, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.grie_rosen_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 4e3

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.happycat_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e3

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.schwefel_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 4e3

		i = 3
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.escaffer6_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 2e7

		i = 4
		tOs = Os[i]
		tMr = Mr[i]
		fit[i] = self.ellips_func(population, nx, tOs, tMr, 1, r_flag)
		fit[i] = 10000 * fit[i] / 1e10

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf07(self, population, nx, Os, Mr, SS, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 3
		delta 	= np.array([10, 30, 50])
		bias 	= np.arange(0, 300, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		tSS = SS[i]
		fit[i] = self.hf01(population, nx, tOs, tMr, tSS, 1, r_flag)

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		tSS = SS[i]
		fit[i] = self.hf02(population, nx, tOs, tMr, tSS, 1, r_flag)

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		tSS = SS[i]
		fit[i] = self.hf03(population, nx, tOs, tMr, tSS, 1, r_flag)

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit
	@classmethod
	def cf08(self, population, nx, Os, Mr, SS, r_flag):
		pop_num = population.shape[0]
		cf_num 	= 3
		delta 	= np.array([10, 30, 50])
		bias 	= np.arange(0, 300, 100).astype('float')

		fit = np.zeros(shape = (cf_num, pop_num))

		i = 0
		tOs = Os[i]
		tMr = Mr[i]
		tSS = SS[i]
		fit[i] = self.hf04(population, nx, tOs, tMr, tSS, 1, r_flag)

		i = 1
		tOs = Os[i]
		tMr = Mr[i]
		tSS = SS[i]
		fit[i] = self.hf05(population, nx, tOs, tMr, tSS, 1, r_flag)

		i = 2
		tOs = Os[i]
		tMr = Mr[i]
		tSS = SS[i]
		fit[i] = self.hf06(population, nx, tOs, tMr, tSS, 1, r_flag)

		final_fit = np.zeros(pop_num)
		for i in range(pop_num):
			final_fit[i] = cf_cal(population[i], nx, Os, delta, bias, fit[:, i], cf_num)
		return final_fit


class CEC14RP_F01(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F01, self).__init__(1, dim_num)

	def evaluate(self, population):
		fit = self.ellips_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 100.

class CEC14RP_F02(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F02, self).__init__(2, dim_num)

	def evaluate(self, population):
		fit = self.bent_cigar_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 200.

class CEC14RP_F03(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F03, self).__init__(3, dim_num)

	def evaluate(self, population):
		fit = self.discus_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 300.

class CEC14RP_F04(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F04, self).__init__(4, dim_num)

	def evaluate(self, population):
		fit = self.rosenbrock_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 400.

class CEC14RP_F05(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F05, self).__init__(5, dim_num)

	def evaluate(self, population):
		fit = self.ackley_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 500.

class CEC14RP_F06(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F06, self).__init__(6, dim_num)

	def evaluate(self, population):
		fit = self.weierstrass_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 600.

class CEC14RP_F07(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F07, self).__init__(7, dim_num)

	def evaluate(self, population):
		fit = self.griewank_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 700.

class CEC14RP_F08(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F08, self).__init__(8, dim_num)

	def evaluate(self, population):
		fit = self.rastrigin_func(population, self.dim_num, self.OShift, self.M, 1, 0)
		return fit + 800.

class CEC14RP_F09(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F09, self).__init__(9, dim_num)

	def evaluate(self, population):
		fit = self.rastrigin_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 900.

class CEC14RP_F10(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F10, self).__init__(10, dim_num)

	def evaluate(self, population):
		fit = self.schwefel_func(population, self.dim_num, self.OShift, self.M, 1, 0)
		return fit + 1000.

class CEC14RP_F11(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F11, self).__init__(11, dim_num)

	def evaluate(self, population):
		fit = self.schwefel_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 1100.

class CEC14RP_F12(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F12, self).__init__(12, dim_num)

	def evaluate(self, population):
		fit = self.katsuura_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 1200.

class CEC14RP_F13(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F13, self).__init__(13, dim_num)

	def evaluate(self, population):
		fit = self.happycat_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 1300.

class CEC14RP_F14(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F14, self).__init__(14, dim_num)

	def evaluate(self, population):
		fit = self.hgbat_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 1400.

class CEC14RP_F15(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F15, self).__init__(15, dim_num)

	def evaluate(self, population):
		fit = self.grie_rosen_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 1500.

class CEC14RP_F16(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F16, self).__init__(16, dim_num)

	def evaluate(self, population):
		fit = self.escaffer6_func(population, self.dim_num, self.OShift, self.M, 1, 1)
		return fit + 1600.

class CEC14RP_F17(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F17, self).__init__(17, dim_num)

	def evaluate(self, population):
		fit = self.hf01(population, self.dim_num, self.OShift, self.M, self.SS, 1, 1)
		return fit + 1700.

class CEC14RP_F18(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F18, self).__init__(18, dim_num)

	def evaluate(self, population):
		fit = self.hf02(population, self.dim_num, self.OShift, self.M, self.SS, 1, 1)
		return fit + 1800.
class CEC14RP_F19(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F19, self).__init__(19, dim_num)

	def evaluate(self, population):
		fit = self.hf03(population, self.dim_num, self.OShift, self.M, self.SS, 1, 1)
		return fit + 1900.
class CEC14RP_F20(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F20, self).__init__(20, dim_num)

	def evaluate(self, population):
		fit = self.hf04(population, self.dim_num, self.OShift, self.M, self.SS, 1, 1)
		return fit + 2000.
class CEC14RP_F21(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F21, self).__init__(21, dim_num)

	def evaluate(self, population):
		fit = self.hf05(population, self.dim_num, self.OShift, self.M, self.SS, 1, 1)
		return fit + 2100.
class CEC14RP_F22(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F22, self).__init__(22, dim_num)

	def evaluate(self, population):
		fit = self.hf06(population, self.dim_num, self.OShift, self.M, self.SS, 1, 1)
		return fit + 2200.
class CEC14RP_F23(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F23, self).__init__(23, dim_num)

	def evaluate(self, population):
		fit = self.cf01(population, self.dim_num, self.OShift, self.M, 1)
		return fit + 2300.
class CEC14RP_F24(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F24, self).__init__(24, dim_num)

	def evaluate(self, population):
		fit = self.cf02(population, self.dim_num, self.OShift, self.M, 1)
		return fit + 2400.
class CEC14RP_F25(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F25, self).__init__(25, dim_num)

	def evaluate(self, population):
		fit = self.cf03(population, self.dim_num, self.OShift, self.M, 1)
		return fit + 2500.
class CEC14RP_F26(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F26, self).__init__(26, dim_num)

	def evaluate(self, population):
		fit = self.cf04(population, self.dim_num, self.OShift, self.M, 1)
		return fit + 2600.
class CEC14RP_F27(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F27, self).__init__(27, dim_num)

	def evaluate(self, population):
		fit = self.cf05(population, self.dim_num, self.OShift, self.M, 1)
		return fit + 2700.
class CEC14RP_F28(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F28, self).__init__(28, dim_num)

	def evaluate(self, population):
		fit = self.cf06(population, self.dim_num, self.OShift, self.M, 1)
		return fit + 2800.
class CEC14RP_F29(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F29, self).__init__(29, dim_num)

	def evaluate(self, population):
		fit = self.cf07(population, self.dim_num, self.OShift, self.M, self.SS, 1)
		return fit + 2900.
class CEC14RP_F30(cec_14_rp_benchmark):
	def __init__(self, dim_num):
		super(CEC14RP_F30, self).__init__(30, dim_num)

	def evaluate(self, population):
		fit = self.cf08(population, self.dim_num, self.OShift, self.M, self.SS, 1)
		return fit + 3000.

########################################################
#				   Helpler Functions				   #
########################################################

def srfunc(population, dim_num, Os, Mr, sh_rate, s_flag, r_flag):
	if s_flag == 1:
		if r_flag == 1:
			population = population - Os
			population *= sh_rate
			population = np.dot(population, Mr.T)
			#population = rotatefunc(population, Mr)
		else:
			population = population - Os
			population *= sh_rate
	else:
		if r_flag == 1:
			population *= sh_rate
			population = np.dot(population, Mr.T)
		else:
			population *= sh_rate
	return population

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

def cf_cal(x, dim_num, Os, delta, bias, fits, cf_num):
	w = np.zeros(cf_num)
	for i in range(cf_num):
		fits[i] += bias[i]
		tmp = np.sum(np.abs(x - Os[i]))
		if tmp == 0.:
			w[i] = float('inf')
		else:
			w[i] = 1. / np.sqrt(np.sum((x - Os[i])**2)) * np.exp(- np.sum((x - Os[i])**2) / (2. * dim_num * (delta[i] ** 2)))
	if np.sum(np.isinf(w)) != 0:
		inf_mask = np.isinf(w)
		fin_mask = np.isfinite(w)
		w[inf_mask] = 1.
		w[fin_mask] = 0.
	if np.sum(np.abs(w)) == 0.:
		w = np.ones(cf_num)
	w = w / np.sum(w)
	return np.sum(w * fits)