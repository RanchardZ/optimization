import os, sys
import numpy as np 
from copy import copy
from constant import CODE_PATH
from benchmark import Benchmark, check_shape



dataFilePath  = os.path.join(CODE_PATH, 'benchmark_funcs', 'CEC_2013_LSOP_Data')
dataFileNames = ['f00.mat', 'f01.mat', 'f02.mat', 'f03.mat', 'f04.mat', 'f05.mat', 'f06.mat', 'f07.mat',
				 'f08.mat', 'f09.mat', 'f10.mat', 'f11.mat', 'f12.mat', 'f13.mat', 'f14.mat', 'f15.mat']
import scipy.io as sio


class cec_13_lsop_benchmark(Benchmark):
	def __init__(self, dim_num, bound, maximize = False):
		super(cec_13_lsop_benchmark, self).__init__(dim_num, bound, maximize)
		self.init_params()
	def init_params(self):
		raise NotImplementedError

###########################################################################################################
# Benchmark Functions for the CEC'2013 Special Session and Compitition on Large-Scale Global Optimization #
###########################################################################################################

######################Base Function#####################

########################################################
#                    Sphere Function                   #
########################################################
def sphere(mat):
	#mat = check_shape(mat)
	fit = np.sum(mat**2, axis = 1)
	return fit

########################################################
#                   Elliptic Function                  #
########################################################
def elliptic(mat):
	#mat = check_shape(mat)
	ps, D = mat.shape
	condition = 1E06
	coeffs 	= np.power(condition, np.linspace(0, 1, D))
	fit = np.sum(coeffs * t_osz_m(mat) ** 2, axis = 1)
	return fit

########################################################
#                 Rastrigin's Function                 #
########################################################
def rastrigin(mat):
	#mat = check_shape(mat)
	ps, D = mat.shape
	A = 10.
	mat_x = t_diag_m(t_asy_m(t_osz_m(mat), 0.2), 10)
	fit = A * (D - np.sum(np.cos(2 * np.pi * mat_x), axis = 1)) +\
	          np.sum(mat_x**2, axis = 1)
	return fit

########################################################
#                   Ackley's Function                  #
########################################################
def ackley(mat):
	#mat = check_shape(mat)
	ps, D = mat.shape
	mat_x = t_diag_m(t_asy_m(t_osz_m(mat), 0.2), 10)
	#mat_x = mat
	fit = np.sum((mat_x/np.sqrt(D))**2, axis = 1)
	fit = 20. - 20.*np.exp(-0.2 * np.sqrt(fit )) - np.exp(np.sum(np.cos(2 * np.pi * mat_x), axis = 1) / D) + np.e
	return fit

########################################################
#                Schwefel's Problem 1.2                #
########################################################
def schwefel(mat):
	ps, D = mat.shape
	mat_x = t_asy_m(t_osz_m(mat), 0.2)
	fit = 0.
	temp_sum = 0.
	for i in range(D):
		temp_sum += mat_x[:, i]
		fit += temp_sum**2
	return fit
########################################################
#               Rosenbrock's Problem 1.2               #
########################################################
def rosenbrock(mat):
	#mat = check_shape(mat)
	ps, D = mat.shape
	fit = np.sum(100. * (mat[:, : -1]**2 - mat[:, 1:])**2 + (mat[:, : -1] - 1)**2, axis = 1)
	return fit

################Fully-Sperable Function#################

########################################################
#           F1: Shifted Elliptic Function              #
########################################################
class CEC13LSOP_F01(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F01, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y 	= check_shape(population) - self.global_optimum
		fit = elliptic(y)
		return fit

	def init_params(self):
		self.data 			= sio.loadmat(os.path.join(dataFilePath, dataFileNames[1]))['xopt']
		self.global_optimum = self.data[:self.dim_num].reshape(1, self.dim_num)

########################################################
#          F2: Shifted Rastrigin's Function            #
########################################################
class CEC13LSOP_F02(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-5., 5.]):
		super(CEC13LSOP_F02, self).__init__(dim_num, bound, False)
		
	def evaluate(self, population):
		y 	= check_shape(population) - self.global_optimum
		fit = rastrigin(y)
		return fit

	def init_params(self):
		self.data 			= sio.loadmat(os.path.join(dataFilePath, dataFileNames[2]))['xopt']
		self.global_optimum = self.data[:self.dim_num].reshape(1, self.dim_num)

########################################################
#           F3: Shifted Ackley's Function              #
########################################################
class CEC13LSOP_F03(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-32., 32.]):
		super(CEC13LSOP_F03, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y 	= check_shape(population) - self.global_optimum
		fit = ackley(y)
		return fit

	def init_params(self):
		self.data 				= sio.loadmat(os.path.join(dataFilePath, dataFileNames[3]))['xopt']
		self.global_optimum 	= self.data[:self.dim_num].reshape(1, self.dim_num)

########################################################
#######Partially Additive Seperable Functions I#########
########################################################

################################################################################################
#           F4: 7-nonseparable, 1-separable Shifted and Rotated Elliptic Function              #
################################################################################################
class CEC13LSOP_F04(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F04, self).__init__(dim_num, bound, False)
		
	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit 		= 0.
		acm 		= 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * elliptic(z_i)
			acm += s
		y_S = y[:, p[acm:] - 1]
		z_S = y_S
		fit += elliptic(z_S)
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[4]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)

################################################################################################
#         F5: 7-nonseparable, 1-separable Shifted and Rotated Rastrigin's Function             #
################################################################################################
class CEC13LSOP_F05(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-5., 5.]):
		super(CEC13LSOP_F05, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * rastrigin(z_i)
			acm += s
		y_S = y[:, p[acm:] - 1]
		z_S = y_S
		fit += rastrigin(z_S)
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[5]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)

################################################################################################
#           F6: 7-nonseparable, 1-separable Shifted and Rotated Ackley's Function              #
################################################################################################
class CEC13LSOP_F06(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-32., 32.]):
		super(CEC13LSOP_F06, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * ackley(z_i)
			acm += s
		y_S = y[:, p[acm:] - 1]
		z_S = y_S
		fit += ackley(z_S)
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[6]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)

################################################################################################
#          F7: 7-nonseparable, 1-separable Shifted and Rotated Schwefel's Function             #
################################################################################################
class CEC13LSOP_F07(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F07, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * schwefel(z_i)
			acm += s
		y_S = y[:, p[acm:] - 1]
		z_S = y_S
		fit += sphere(z_S)
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[7]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)

########################################################
#######Partially Additive Seperable Functions II########
########################################################
################################################################################################
#                 F8: 20-nonseparable Shifted and Rotated Elliptic Function                    #
################################################################################################
class CEC13LSOP_F08(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F08, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * elliptic(z_i)
			acm += s
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[8]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)
################################################################################################
#                F9: 20-nonseparable Shifted and Rotated Rastrigin's Function                  #
################################################################################################
class CEC13LSOP_F09(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-5., 5.]):
		super(CEC13LSOP_F09, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * rastrigin(z_i)
			acm += s
		return fit 

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[9]))
		self.global_optimum = self.data['xopt'][:self.dim_num].reshape(1, self.dim_num)
################################################################################################
#                 F10: 20-nonseparable Shifted and Rotated Ackley's Function                   #
################################################################################################
class CEC13LSOP_F10(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-32., 32.]):
		super(CEC13LSOP_F10, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * ackley(z_i)
			acm += s
		return fit 

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[10]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)

################################################################################################
#                 F11: 20-nonseparable Shifted and Rotated Schwefel's Function                 #
################################################################################################
class CEC13LSOP_F11(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F11, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.	
		acm = 0
		for w_i, s in zip(self.data['w'], self.data['s']):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = y[:, p[acm: acm+s] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * schwefel(z_i)
			acm += s
		return fit 

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[11]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)

########################################################
#################Overlapping Functions##################
########################################################
################################################################################################
#                             F12: shifted Rosenbrock's Function                               #
################################################################################################
class CEC13LSOP_F12(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F12, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		population = check_shape(population)
		y = population - self.global_optimum
		fit = rosenbrock(y)
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[12]))
		self.global_optimum = self.data['xopt'][:self.dim_num].reshape(1, self.dim_num)

################################################################################################
#           F13: Shifted Schwefel's Function with Conforming Overlapping Subcomponents         #
################################################################################################
class CEC13LSOP_F13(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 905, bound = [-100., 100.]):
		super(CEC13LSOP_F13, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		y = check_shape(population) - self.global_optimum
		p = self.data['p'].flatten().astype(int)
		fit = 0.
		acm = 0
		for i, (w_i, s) in enumerate(zip(self.data['w'], self.data['s'])):
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			#print '[%d, %d]' % (acm - i*self.m, acm + s - i*self.m)
			y_i = y[:, p[acm - i*self.m: acm + s - i*self.m] - 1].transpose()
			z_i = np.dot(R, y_i).transpose()
			#print w_i * schwefel(z_i)
			fit += w_i * schwefel(z_i)
			acm += s
		return fit
	def init_params(self):
		self.data 			= sio.loadmat(os.path.join(dataFilePath, dataFileNames[13]))
		self.global_optimum = self.data['xopt'][:self.dim_num].reshape(1, self.dim_num)
		self.m 				= int(self.data['m'][0][0])
################################################################################################
#          F14: Shifted Schwefel's Function with Conflicting Overlapping Subcomponents         #
################################################################################################
class CEC13LSOP_F14(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 905, bound = [-100., 100.]):
		super(CEC13LSOP_F14, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		x = check_shape(population)
		p = self.data['p'].flatten()
		p = p.astype(int)
		fit = 0.
		acm = 0
		for i, (w_i, s) in enumerate(zip(self.data['w'], self.data['s'])):
			#print i,
			s = int(s[0])
			if s == 25:
				R = self.data['R25']
			elif s == 50:
				R = self.data['R50']
			else:
				R = self.data['R100']
			y_i = (x[:, p[acm - i*self.m: acm + s - i*self.m] - 1] -\
						self.global_optimum[:, acm: acm+s]).transpose()
			z_i = np.dot(R, y_i).transpose()
			fit += w_i * schwefel(z_i)
			acm += s
		return fit

	def init_params(self):
		self.data 			= sio.loadmat(os.path.join(dataFilePath, dataFileNames[14]))
		self.global_optimum = self.data['xopt'].reshape(1, 1000)
		self.m 				= int(self.data['m'][0][0])

################################################################################################
#                              F15: Shifted Schwefel's Function                                #
################################################################################################
class CEC13LSOP_F15(cec_13_lsop_benchmark):
	def __init__(self, dim_num = 1000, bound = [-100., 100.]):
		super(CEC13LSOP_F15, self).__init__(dim_num, bound, False)

	def evaluate(self, population):
		z = check_shape(population) - self.global_optimum
		fit = schwefel(z)
		return fit

	def init_params(self):
		self.data = sio.loadmat(os.path.join(dataFilePath, dataFileNames[15]))
		self.global_optimum = self.data['xopt'][: self.dim_num].reshape(1, self.dim_num)


########################################################
#					Helpler Functions				   #
########################################################
def t_osz(vec):
	divisor = .1
	new_vec = copy(vec)
	zero_more = vec > 0
	new_vec[zero_more] = np.log(vec[zero_more]) / divisor
	new_vec[zero_more] = np.power(np.exp(new_vec[zero_more] + 0.49 * (np.sin(new_vec[zero_more]) + np.sin(0.79 * new_vec[zero_more]))), divisor)
	zero_less = vec < 0
	new_vec[zero_less] = np.log(-vec[zero_less]) / divisor
	new_vec[zero_less] = -np.power(np.exp(new_vec[zero_less] + 0.49 * (np.sin(0.55 * new_vec[zero_less]) + np.sin(0.31 * new_vec[zero_less]))), divisor)
	return new_vec

def t_osz_m(mat):
	divisor = .1
	new_mat = copy(mat)
	zero_more = mat > 0
	new_mat[zero_more] = np.log(mat[zero_more]) / divisor
	#new_mat[zero_more] = np.power(np.exp(new_mat[zero_more] + 0.49 * (np.sin(new_mat[zero_more]) + np.sin(0.79 * new_mat[zero_more]))), divisor)
	new_mat[zero_more] = np.power(np.exp(new_mat[zero_more] + 0.049 * (np.sin(10. * new_mat[zero_more]) + np.sin(7.9 * new_mat[zero_more]))), divisor)
	zero_less = mat < 0
	new_mat[zero_less] = np.log(-mat[zero_less]) / divisor
	#new_mat[zero_less] = -np.power(np.exp(new_mat[zero_less] + 0.49 * (np.sin(0.55 * new_mat[zero_less]) + np.sin(0.31 * new_mat[zero_less]))), divisor)
	new_mat[zero_less] = -np.power(np.exp(new_mat[zero_less] + 0.049 * (np.sin(5.5 * new_mat[zero_less]) + np.sin(3.1 * new_mat[zero_less]))), divisor)
	return new_mat

def t_asy(vec, beta):
	D = len(vec)
	new_vec = copy(vec)
	temp = beta * np.linspace(0, 1, D)
	zero_more = vec > 0
	new_vec[zero_more] = np.power(vec[zero_more], 1 + temp[zero_more] * np.sqrt(vec[zero_more]))
	return new_vec

def t_asy_m(mat, beta):
	popsize, D = mat.shape
	new_mat = copy(mat)
	temp = np.repeat(beta * np.linspace(0, 1, D).reshape(1, D), popsize, axis = 0)
	zero_more = mat > 0
	new_mat[zero_more] = np.power(mat[zero_more], 1 + temp[zero_more] * np.sqrt(mat[zero_more]))
	return new_mat

def t_diag(vec, alpha):
	D = len(vec)
	new_vec = np.diag(np.power(np.sqrt(alpha), np.linspace(0, 1, D))) * vec
	return new_vec

def t_diag_m(mat, alpha):
	D = mat.shape[1]
	#new_mat = np.diag(np.power(np.sqrt(alpha), np.linspace(0, 1, D))) * mat
	new_mat = np.power(np.sqrt(alpha), np.linspace(0, 1, D)) * mat
	return new_mat