import os, sys
import numpy as np 
from cec_14_real_parameter import *
from constant import CODE_PATH

input_data_path 	= os.path.join(CODE_PATH, 'benchmark_funcs', 'CEC_2014_RP_Data')

func_num = 30
dim_num = 10

x = np.zeros(shape=(3, dim_num))
#print x
x[2] = np.ones(dim_num) * 10.
#print "b = CEC14RP_F%02d(dim_num)" % func_num
#exec("b = CEC14RP_F%02d(dim_num)" % func_num)
#print b.OShift
#print b.evaluate(x)

for i in range(1, func_num+1):
	x[0] = map(float, open(os.path.join(input_data_path, "shift_data_%d.txt" % i)).readline().strip().split())[: dim_num]	
	#print "b = CEC14RP_F%02d(dim_num)" % i
	exec("b%02d = CEC14RP_F%02d(dim_num)" % (i, i))
	exec("fit = b%02d.evaluate(x)" % i)
	#fit = b.evaluate(x)

	for j in range(3):
		print "f{0}(x[{1}])={2}".format(i, j, fit[j])
	#print fit

