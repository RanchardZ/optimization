import time

class Individual(object):

	def __init__(self, feature = None, value = None):
		self.feature 	= feature
		self.value 		= value
		self.birthdate 	= time.time()

	def __setattr__(self, name, val):
		if name == 'feature':
			self.value = None
		self.__dict__[name] = val

	def __str__(self):
		return '{0} : {1}'.format(self.feature, self.value)

	def __repr__(self):
		return '<Feature = {0}, value = {1}, birthdate = {2}>'\
				.format(self.feature, self.value, self.birthdate)

	def __lt__(self, other):
		if self.value != None and other.value != None:
			return self.value > other.value
		else:
			raise Exception('value of Individual can not be None')

	def __gt__(self, other):
		if self.value != None and other.value != None:
			return other < self
		else:
			raise Exception('value of Individual can not be None')

	def __le__(self, other):
		return self < other or not other < self

	def __ge__(self, other):
		return other < self or not self < other

	def __eq__(self, other):
		return self.feature == other.feature

	def __ne__(self, other):
		return self.feature != self.feature
		