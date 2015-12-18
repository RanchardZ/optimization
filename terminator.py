

class Terminator(object):
	def __init__(self, **kwargs):
		self.n_iter 		= kwargs.setdefault('num_iterations', 0)
		self.max_iter 		= kwargs.get('max_iterations')
		self.n_eval 		= kwargs.setdefault('num_evaluations', 0)
		self.max_eval 		= kwargs.get('max_evaluations')
		self.t_type			= kwargs.setdefault('terminator_type', 'iteration_terminator')
		self.source 		= kwargs.get('source')
		self.cur_diversity 	= kwargs.setdefault('diversity', float('inf'))
		self.min_diversity	= kwargs.get('min_diversity')

	def should_terminate(self):
		if self.t_type == 'iteration_terminator':
			return shouldTerminateIteration(num_iterations = self.n_iter,
											  max_iterations = self.max_iter)
		elif self.t_type == 'evaluation_terminator':
			return shouldTerminateEvaluation(num_evaluations = self.n_eval,
											   max_evaluations = self.max_eval)
		elif self.t_type == 'default_terminator':
			return shouldTerminateDefault(num_evaluations = self.n_eval,
											   max_evaluations = self.max_eval)

	def add_stat(self, **kwargs):
		iter_inc = kwargs.setdefault('iter_inc', 0)
		eval_inc = kwargs.setdefault('eval_inc', 0)
		self.n_iter += iter_inc
		self.n_eval += eval_inc

	def update_diversity(self, diversity):
		self.cur_diversity = diversity

	def get_stat(self):
		return self.n_iter, self.n_eval, self.cur_diversity

	def get_message(self):
		if self.source == None:
			raise Exception('unknown source of terminator')
		if self.t_type == 'iteration_terminator':
			return '<iteration {0:^6}>'.format(self.n_iter), self.source
		elif self.t_type == 'evaluation_terminator':
			return '<evaluation {0:^8d}>'.format(self.n_eval), self.source


def shouldTerminate(fn):
	def inner(**kwargs):
		return fn(**kwargs)
	return inner

@shouldTerminate
def shouldTerminateDefault(**kwargs):
	return True

@shouldTerminate
def shouldTerminateIteration(**kwargs):
	num_iterations = kwargs.get('num_iterations')
	max_iterations = kwargs.get('max_iterations')
	if num_iterations == None or max_iterations == None:
		raise Exception('num_iterations or max_iterations not defined.')
	return num_iterations >= max_iterations

@shouldTerminate
def shouldTerminateEvaluation(**kwargs):	
	num_evaluations = kwargs.get('num_evaluations')
	max_evaluations = kwargs.get('max_evaluations')
	if num_evaluations == None or max_evaluations == None:
		raise Exception('num_evaluations or max_evaluations not defined.')
	return num_evaluations >= max_evaluations

@shouldTerminate
def shouldTerminateDiversity(**kwargs):
	cur_diversity 	= kwargs.get('diversity')
	min_diversity 	= kwargs.get('min_diversity')
	if cur_diversity == None or min_diversity == None:
		raise Exception('diversity or min_diversity not defined')
	return cur_diversity < min_diversity
