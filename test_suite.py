from common import *

karaboga_benchmarks 				= ['Rosenbrock', 'Rastrigin', 'Sphere', 'Griewank', 'Ackley']
karaboga_dim_nums 					= [50, 50, 100, 50, 200]
karaboga_max_evals 					= 5E4
karaboga_observe_points 			= range(10000, 60000, 10000)

cec_13_lsop_benchmarks				= ['CEC13LSOP_F01', 'CEC13LSOP_F02', 'CEC13LSOP_F03', 'CEC13LSOP_F04',\
									   'CEC13LSOP_F05', 'CEC13LSOP_F06', 'CEC13LSOP_F07', 'CEC13LSOP_F08',\
									   'CEC13LSOP_F09', 'CEC13LSOP_F10', 'CEC13LSOP_F11', 'CEC13LSOP_F12',\
									   'CEC13LSOP_F13', 'CEC13LSOP_F14', 'CEC13LSOP_F15']
cec_13_lsop_dim_nums 				= [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,\
									   1000, 1000, 1000, 1000, 905, 905, 1000]
cec_13_lsop_max_evals 				= 3E6
cec_13_lsop_observe_points 			= range(100000, 3100000, 100000)

cec_13_real_parameter_benchmarks 	= ['CEC13RP_F01', 'CEC13RP_F02', 'CEC13RP_F03', 'CEC13RP_F04', 'CEC13RP_F05',\
									   'CEC13RP_F06', 'CEC13RP_F07', 'CEC13RP_F08', 'CEC13RP_F09', 'CEC13RP_F10',\
									   'CEC13RP_F11', 'CEC13RP_F12', 'CEC13RP_F13', 'CEC13RP_F14', 'CEC13RP_F15',\
									   'CEC13RP_F16', 'CEC13RP_F17', 'CEC13RP_F18', 'CEC13RP_F19', 'CEC13RP_F20',\
									   'CEC13RP_F21', 'CEC13RP_F22', 'CEC13RP_F23', 'CEC13RP_F24', 'CEC13RP_F25',\
									   'CEC13RP_F26', 'CEC13RP_F27', 'CEC13RP_F28']
cec_13_real_parameter_dim_nums  	= (np.ones(30) * 30).astype('int').tolist()
cec_13_real_parameter_max_evals 	= 5E5
cec_13_real_parameter_observe_points = range(100000, 600000, 100000)

cec_14_real_parameter_benchmarks 	= ['CEC14RP_F01', 'CEC14RP_F02', 'CEC14RP_F03', 'CEC14RP_F04', 'CEC14RP_F05',\
									   'CEC14RP_F06', 'CEC14RP_F07', 'CEC14RP_F08', 'CEC14RP_F09', 'CEC14RP_F10',\
									   'CEC14RP_F11', 'CEC14RP_F12', 'CEC14RP_F13', 'CEC14RP_F14', 'CEC14RP_F15',\
									   'CEC14RP_F16', 'CEC14RP_F17', 'CEC14RP_F18', 'CEC14RP_F19', 'CEC14RP_F20',\
									   'CEC14RP_F21', 'CEC14RP_F22', 'CEC14RP_F23', 'CEC14RP_F24', 'CEC14RP_F25',\
									   'CEC14RP_F26', 'CEC14RP_F27', 'CEC14RP_F28', 'CEC14RP_F29', 'CEC14RP_F30']
cec_14_real_parameter_dim_nums  	= (np.ones(30) * 30).astype('int').tolist()
cec_14_real_parameter_max_evals 	= 3E5
cec_14_real_parameter_observe_points = range(10000, 310000, 10000)

cec_05_real_parameter_benchmarks  	= ['CEC05RP_F01', 'CEC05RP_F02', 'CEC05RP_F03', 'CEC05RP_F04', 'CEC05RP_F05',\
									   'CEC05RP_F06', 'CEC05RP_F07', 'CEC05RP_F08', 'CEC05RP_F09', 'CEC05RP_F10',\
									   'CEC05RP_F11', 'CEC05RP_F12', 'CEC05RP_F13', 'CEC05RP_F14', 'CEC05RP_F15',\
									   'CEC05RP_F16', 'CEC05RP_F17', 'CEC05RP_F18', 'CEC05RP_F19', 'CEC05RP_F20',\
									   'CEC05RP_F21', 'CEC05RP_F22', 'CEC05RP_F23', 'CEC05RP_F24', 'CEC05RP_F25']
cec_05_real_parameter_dim_nums 		= (np.ones(25) * 30).astype('int').tolist()
cec_05_real_parameter_max_evals 	= 3E5
cec_05_real_parameter_observe_points = range(10000, 310000, 10000)

class Suite(object):

	def __init__(self, suite, pop_nums):		
		exec("self.benchmarks = %s_benchmarks" % suite)
		exec("self.dim_nums = %s_dim_nums" % suite)
		self.pop_nums = pop_nums
		assert(len(self.benchmarks) == len(pop_nums))
		self.benchmark2pop_num = dict(zip(self.benchmarks, pop_nums))
		self.benchmark2dim_num = dict(zip(self.benchmarks, self.dim_nums))

	def get_pop_num(self, benchmark):
		return self.benchmark2pop_num[benchmark]

	def get_dim_num(self, benchmark):
		return self.benchmark2dim_num[benchmark]

		

def algorithm_test(package, algorithm, pop_nums, suite, epoch_num, archive_switch):
	exec('from %s import %s' % (package, algorithm))
	ts 				= Suite(suite = suite, pop_nums = pop_nums)
	project_name 	= 'suite_test'
	experiment_name = suite
	total_done_count = 0
	total_undone_count = 0
	for target in ts.benchmarks:
		pop_num = ts.get_pop_num(target)
		dim_num = ts.get_dim_num(target)
		exec("max_evl = %s_max_evals" % suite)
		done_count = 0
		undone_count = 0
		for epoch in range(1, epoch_num + 1):
			stat_file = os.path.join(DATA_PATH, project_name, experiment_name, 'rawData',\
								 '_'.join([algorithm, str(int(pop_num)), str(dim_num), target, str(int(max_evl)), str(epoch)])) + '.sto'
			if os.path.exists(stat_file):
				done_count += 1
				continue
			else:
				undone_count += 1
				exec('spe = %s(pop_num, dim_num, target, max_evaluations = %s_max_evals, observe_points = %s_observe_points,\
							   archive_switch = archive_switch, project_name = project_name, info_switch = False,\
							   experiment_name = experiment_name, epoch = epoch)' % (algorithm, suite, suite))
				spe.evolveSpecies()
				
		print "%s: %d done, %d undone" % (target, done_count, undone_count)
		total_done_count += done_count
		total_undone_count += undone_count

	print "%d done" % total_done_count
	print "%d undone" % total_undone_count

def single_algorithm_plot(package, algorithm, pop_num, dim_num, benchmark_name, max_evaluations, epoch_num, archive_switch):
	exec('from %s import %s' % (package, algorithm))

	project_name 	= 'plot'
	experiment_name = 'single_algorithm_plot'

	for epoch in range(1, epoch_num + 1):
		if os.path.exists(os.path.join(DATA_PATH, project_name, experiment_name, 'rawData', '_'.join([algorithm, str(pop_num), str(dim_num), benchmark_name, str(epoch)]) + '.sto')):
			continue
		else:
			exec('spe = %s(pop_num, dim_num, benchmark_name, max_evaluations = max_evaluations, info_switch = False,\
						   archive_switch = archive_switch, project_name = project_name, experiment_name = experiment_name,\
						   epoch = epoch)' % (algorithm))
			spe.evolveSpecies()
	generate_head_files(os.path.join(DATA_PATH, project_name, experiment_name, 'rawData'))
	generate_stat_files(project_name, experiment_name)


def portfolio_composition(portfolio_pkgs, portfolio_algs, portfolio_pop_nums, epoch_num, max_evals, suite):

	alg_num = len(portfolio_algs)
	exec('ben_num = len(%s_benchmarks)' % suite)

	try:
		len(portfolio_pop_nums)
		assert(len(portfolio_pop_nums) == ben_num)
	except:
		portfolio_pop_nums = [portfolio_pop_nums] * ben_num
	
	for pkg, alg, pop_num in zip(portfolio_pkgs, portfolio_algs, portfolio_pop_nums):
		algorithm_test(pkg, alg, pop_num, suite, epoch_num, archive_switch = True)
	
	# create stat_files_dirs and find the best_scores
	evals = np.zeros(ben_num, alg_num)
	errors = np.zeros(ben_num, alg_num)
	scores = np.zeros(ben_num, alg_num)
	stat_files_dirs = [[None for j in range(alg_num)] for i in range(ben_num)]
	for i in range(ben_num):
		for j in range(alg_num):
			exec("dim_num = %s[i]" % '_'.join([suite, 'dim_nums']))
			exec('ben_name = %s_benchmarks[i]' % suite)
			stat_files_dirs[i][j] = '_'.join([portfolio_algs[j], str(portfolio_pop_nums[i]), str(dim_num), ben_name])
			# figure out the ranking here!!!!
			n_i, n_e, av, stav, gv, stgv, ad, stad = read_stat_file('suite_test', suite, stat_files_dirs[i][j])			
			exec('min_fitness = %s_benchmarks[i](dim_num).min_fitness' % suite)
			idx = 0
			while abs(gv[idx] - min_fitness) > 1E-6:
				idx += 1
			if idx == len(n_i):
				evals[i, j] = n_e[idx]
			else:
				evals[i, j] = max_evals
			errors[i, j] = gv[-1] - min_fitness
	ranks = alg_rank(evals, errors)
	avg_ranks = np.average(ranks, axis = 0)

	b_alg_idx 	= np.argsort(avg_ranks)[0] 				# index of the best algorithm
	corrcoef 	= np.corrcoef(ranks)
	o_alg_idx 	= np.argsort(corrcoef[b_alg_idx])[0] 	# index of the algorithm that is least correlated with the best algorithm
	return [portfolio_algs[b_alg_idx], portfolio_algs[o_alg_idx]]


def alg_rank(evals, errors):
	""" tested:

		errors = [[0	, 5		, 3  	, 0		],
				  [0	, 100	, 200	, 0		],
				  [0	, 0  	, 0  	, 0		],
				  [100	, 200 	, 500 	, 250	]]
		evals  = [[300	, 500	, 500  	, 100	],
				  [100	, 1000	, 1000	, 800	],
				  [100	, 200  	, 400  	, 300	],
				  [1000	, 1000 	, 1000 	, 1000	]]
		ranks  = [[2	, 4		, 3  	, 1		],
				  [1	, 3		, 4		, 2		],
				  [1	, 2  	, 4  	, 3		],
				  [1	, 2  	, 4  	, 3		]]
	"""
	evals = evals.astype('float')
	errors = errors.astype('float')
	ben_num, alg_num = errors.shape
	ranks = np.zeros_like(evals)

	max_errors = np.max(errors, axis = 1).reshape(ben_num, 1)
	max_errors[max_errors == 0] = 1.
	nor_errors 	= 10. * errors / max_errors
	e_ranks = np.zeros_like(nor_errors)
	s_idxs 	= np.argsort(nor_errors, axis = 1)
	for i, idxs in enumerate(s_idxs):
		e_ranks[i, idxs] = np.arange(alg_num) + 1.
	scores = evals + e_ranks
	s_idxs 	= np.argsort(scores, axis = 1)
	for i, idxs in enumerate(s_idxs):
		ranks[i, idxs] = np.arange(alg_num) + 1.
	return ranks # float type
	


