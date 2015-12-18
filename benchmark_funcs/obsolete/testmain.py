import numpy as np
from testfunc import *

b1 = f1()

x = np.zeros(shape = (3, 10))
x[1] = b1.os[0]
x[2] = np.arange(10).astype('float')
print x

for i in range(1, 29):
    exec('b = f%d()' % i)
    res = b.evaluate(x)
    print 'f{}: {}'.format(i, res)
