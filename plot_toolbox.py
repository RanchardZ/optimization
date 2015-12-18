import os, sys
import numpy as np
from copy import copy
import matplotlib.pyplot as plt


grid = True
sizes   = range(10, 81, 10)
colors  = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
markers = ['o', '+', 'x', 's', '^', 'd', 'h', '8']

class Plot(object):

	def __init__(self, filePath, xlist, ylist, labelList, figName='temp_figure.png', **kwargs):
		self.filePath			= filePath
		self.xlist, self.ylist  = xlist, ylist		
		self.labelList 			= labelList
		self.length				= len(self.xlist)
		self.semilogy 			= kwargs.setdefault('semilogy', 'False')
		self.figName 			= figName

	def line_plot(self):
		for i, (x, y, l) in enumerate(zip(self.xlist, self.ylist, self.labelList)):
			if self.semilogy:
				self.set_logy()
			try:
				plt.plot(x, y, color=colors[i % len(colors)], linewidth=1.0, label=l)
			except:
				printMessage('Plot', 'ERROR', 'Something wrong with the inputs.')

	def scatter_plot(self):
		for i,(x,y) in enumerate(zip(self.xlist, self.ylist)):
			plt.scatter(x, y, color=colors[i % len(colors)])

	def sub_plot(self):
		fig = plt.figure()
		names = ['ax'+str(i+1) for i in range(self.length)]
		index_generator = self._subplot_index()
		for i, (x, y) in enumerate(zip(self.xlist, self.ylist)):
			sub_index = index_generator.next()
			print sub_index
			exec('%s = fig.add_subplot(%s)' % (names[i], sub_index))
			exec('%s.plot(%s, %s)' % (names[i], 'self.xlist[%d]'%i, 'self.ylist[%d]'%i))

	def fill_between(self, lower_line_index, upper_line_index):
		assert(len(self.xlist[upper_line_index]) == len(self.xlist[lower_line_index]))
		x 	= self.xlist[upper_line_index][:]
		y1 	= self.ylist[lower_line_index]
		y2 	= self.ylist[upper_line_index]
		plt.fill_between(x, y2, y1, y2 > y1, color = '#ff0000', alpha = 0.3)

	def show_plot(self):
		plt.show()

	def save_plot(self):
		resultFile = os.path.join(self.filePath, self.figName)
		try:
			os.makedirs(self.filePath)
		except:
			pass
		plt.savefig(resultFile, dpi = 500)

	def close_plot(self):
		plt.close()

	def set_logy(self):
		plt.semilogy()

	def set_title(self, title):
		plt.title(title)

	def set_xlabel(self, xlabel):
		plt.xlabel(xlabel)

	def set_ylabel(self, ylabel):
		plt.ylabel(ylabel)

	def set_legend(self, legend='upper right'):
		#plt.legend(legend)
		plt.legend()

	def _subplot_index(self):
		sub_num = self.length
		size = int(math.ceil(math.sqrt(sub_num)))
		i = 0
		while i < sub_num:
			row = i / size
			col = i % size
			yield '%d%d%d' % (size, size, row*size+col+1)
			i += 1
			
	def _generate_subplots_index(self):
		sub_num = len(self.xlist)
		size = int(math.ceil(math.sqrt(sub_num)))
		return self._subplots_index_generator(size)

	def _subplots_index_generator(self, size):
		for i in range(1, size+1):
			for j in range(1, size+1):
				yield '%d%d%d' % (size, size, (i-1)*size+j)
