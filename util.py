import os, sys, math
import numpy as np



def printMessage(m_source, m_type, m_content):
	print '%s from %s: %s' % (m_type, m_source, m_content)

def rouletteWheelSelection(scores):
	rd_num = np.random.rand() * np.sum(scores)
	select_index = 0
	select = scores[select_index]
	while (select < rd_num) and (select_index < len(scores)):
		select_index 	+= 1
		select 			+= scores[select_index]
	return select_index

def stochasticUniversalSampling(scores, num_to_select):
	total_scores 	= sum(scores)
	select_step		= total_scores / num_to_select
	start 	 		= np.random.rand() * select_step
	pointers 		= np.arange(num_to_select).astype('float') * select_step + start
	select_idxes 	= []
	for pointer in pointers:
		select_idx 	= 0
		select 		= scores[select_idx]
		while (select < pointer) and (select_idx < len(scores)):
			select_idx 	+= 1
			select 		+= scores[select_idx]
		select_idxes.append(select_idx)
	return select_idxes

def makeDirsForFile(fileName):
	try:
		os.makedirs(os.path.split(fileName)[0])
	except:
		printMessage('makeDirsForFile', 'ERROR', 'Failed to make directories for %s.' % os.path.split(fileName)[1])

def makeDirs(dirName):
	try:
		os.makedirs(dirName)
	except:
		printMessage('makeDirs', 'ERROR', 'Failed to make directories for %s.' % dirName)

class cmdOptions(object):
	def __init__(self):
		self.vdict = {}
		self.hdict = {}

		self.addOption('ROOT_PATH', ROOT_PATH, '--rootpath [default: %s]' % ROOT_PATH)
		self.addOption('overwrite', False, '--overwrite [default: False]')
	def addOption(self, key, val, hlp):
		self.vdict[key] = val
		self.hdict[key] = hlp
	def getString(self, key):
		return self.vdict[key]
	def printHelp(self):
		for k, v in self.hdict.items():
			print v
	def parse(self, argv):
		i = 0
		while i < len(argv) -1:
			if argv[i].startswith("--"):
				if argv[i+1].startswith("--"):
					i += 1
					continue
				param = argv[i][2:]
				if param in self.vdict:
					self.vdict[param] = argv[i+1]
					i += 2
				else:
					i += 1
			else:
				i += 1
		okay = self.checkArgs()
		if not okay:
			self.printHelp()
		return okay
	def checkArgs(self):
		paramsNeeded = [key for (key,val) in self.vdict.iteritems() if val is ""]
		if paramsNeeded:
			printMessage(self.__class__.__name__, "ERROR", "Need more arguments: %s" % " ".join(paramsNeeded))
			return False
		return True

