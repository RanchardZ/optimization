import os, sys
import cPickle as pickle
from util import printMessage, makeDirsForFile

class Archiver(object):

	def __init__(self, source, file_path, file_name, archive_switch):
		self.source = source
		self.file_path, self.file_name = file_path, file_name
		self.switch = archive_switch

	def openFileToWrite(self):
		"""open a file and write from the start if it does not exist.
		   open a file and write after EOF if it already exists
		"""
		if self.switch:
			targetFile = os.path.join(self.file_path, self.file_name)
			makeDirsForFile(targetFile)
			self.file = open(targetFile, 'a')
		else:
			return

	def openFileToRead(self):
		"""open only an existing file
		"""
		if self.switch:
			try:
				self.file = open(os.path.join(self.file_path, self.file_name), 'r')
			except:
				printMessage(str(self.__class__), 'Error', 'file does not exist.')
		else:
			return

	def writeToFile(self, data_deliver):
		"""data should be a list of data items. [data_1, data_2, data_3,..., data_n]
		"""
		if self.switch:
			pickle.dump(data_deliver, self.file)
		else:
			return

	def readFromFile(self):
		if self.switch:
			return pickle.load(self.file)
		else:
			return

	def closeFile(self):
		if self.switch:
			self.file.close()
		else:
			return