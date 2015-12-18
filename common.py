import os, sys
import numpy as np
from copy import copy
from archiver import Archiver
from util import printMessage, makeDirs
from plot_toolbox import Plot
from constant import CODE_PATH, DATA_PATH

def generate_head_files(fileDir):
	fileSet			= os.listdir(fileDir)
	preFix2FileName = {}
	for f in fileSet:
		preFix = f[:f.rfind('_')]
		if preFix2FileName.has_key(preFix):
			preFix2FileName[preFix].append(f)
		else:
			preFix2FileName[preFix] = [f]
	
	headFilesPath = os.path.join(os.path.split(fileDir)[0], 'headFiles')
	makeDirs(headFilesPath)

	for name in preFix2FileName:
		ful_name = os.path.join(headFilesPath, name + '.hf')
		if os.path.exists(ful_name):
			continue
		f = open(ful_name, 'w')
		for elem in preFix2FileName[name]:
			f.write(elem + '\n')
		f.close()

def get_head_files(project_name, experiment_name):
	return os.listdir(os.path.join(DATA_PATH, project_name, experiment_name, 'headFiles'))

def read_from_head_file(filePath):
	f = open(filePath)
	data_files = map(str.strip, f.readlines())
	f.close()
	return data_files

def generate_stat_files(project_name, experiment_name):
	statDir = os.path.join(DATA_PATH, project_name, experiment_name)
	try:
		makeDirs(statDir)
	except:
		pass
	headFiles = get_head_files(project_name, experiment_name)
	for headFile in headFiles:
		data_files 	= read_from_head_file(os.path.join(DATA_PATH, project_name, experiment_name, 'headFiles', headFile))
		find_stat(data_files, project_name, experiment_name, headFile) 

def find_stat(dataFiles, project_name, experiment_name, headFile, overwrite = 0):
	statFilePath 		= os.path.join(DATA_PATH, project_name, experiment_name, 'statFiles', headFile[:headFile.rfind('.')])
	makeDirs(statFilePath)
	avg_over_time_file_path = os.path.join(statFilePath, headFile[:headFile.rfind('.')] + '.stat')
	if os.path.exists(avg_over_time_file_path) and (not overwrite):
		return
	avg_over_time_file  	= open(avg_over_time_file_path, 'w')
	aot_iter, aot_eval 	= [], []
	m_length = 100000000000000000 # large number
	for dataFile in dataFiles:
		archiv = Archiver('find_stat', os.path.join(DATA_PATH, project_name, experiment_name, 'rawData'), dataFile, True)
		archiv.openFileToRead()
		num_iter, num_eval, best_value, g_best_value, diversity = [], [], [], [], []
		while True:
			try:
				ni, ne, bv, gbv, dvt = archiv.readFromFile()
				num_iter.append(ni)
				num_eval.append(ne)
				best_value.append(bv)
				g_best_value.append(gbv)
				diversity.append(dvt)
			except:
				printMessage('find_stat', 'INFO', 'Finish reading file %s' % dataFile)
				break
		if len(num_iter) < m_length:
			m_length = len(num_iter)
		print dataFile, m_length

		best_value_arr 		= np.array(best_value)[:m_length]
		g_best_value_arr	= np.array(g_best_value)[:m_length]
		diversity_arr 		= np.array(diversity)[:m_length]

		if len(aot_iter) == 0:
			aot_iter 	= copy(num_iter)
			aot_eval 	= copy(num_eval)
			aot_bv   	= copy(best_value_arr).reshape(1, m_length)
			aot_gbv  	= copy(g_best_value_arr).reshape(1, m_length)
			aot_dvt  	= copy(diversity_arr).reshape(1, m_length)
		else:
			aot_bv 	= np.concatenate((aot_bv[:, :m_length], best_value_arr.reshape(1, m_length)), axis = 0)
			aot_gbv = np.concatenate((aot_gbv[:, :m_length], g_best_value_arr.reshape(1, m_length)), axis = 0)
			aot_dvt	= np.concatenate((aot_dvt[:, :m_length], diversity_arr.reshape(1, m_length)), axis = 0) 

	abv 		= np.average(aot_bv, axis = 0)
	stdbv 		= np.std(aot_bv, axis = 0)
	agbv 		= np.average(aot_gbv, axis = 0) 
	stdgbv 		= np.std(aot_gbv, axis = 0)
	advt 		= np.average(aot_dvt, axis = 0)
	stddvt		= np.std(aot_dvt, axis = 0)

	for i, e, ab, sb, ag, sg, ad, sd in zip(aot_iter, aot_eval, abv, stdbv, agbv, stdgbv, advt, stddvt):
		avg_over_time_file.write(' '.join([str(i), str(e), str(ab), str(sb), str(ag), str(sg), str(ad), str(sd)]) + '\n')
	avg_over_time_file.close()

def read_stat_file(project_name, experiment_name, stat_file_dir):
	print 'reading from: ' + os.path.join(DATA_PATH, project_name, experiment_name, 'statFiles', stat_file_dir)
	statFilePath 	= os.path.join(DATA_PATH, project_name, experiment_name, 'statFiles', stat_file_dir)
	statFile 		= open(os.path.join(statFilePath, stat_file_dir+'.stat'), 'r')
	statData 		= map(str.strip, statFile.readlines())
	num_iter, num_eval, abv, stdbv, agbv, stdgbv, advt, stddvt = [], [], [], [], [], [], [], []
	for data_tuple in statData:
		i, e, ab, sb, ag, sg, ad, sd	= data_tuple.split()
		i, e 							= int(i), int(e)
		ab, sb, ag, sg, ad, sd 		= float(ab), float(sb), float(ag), float(sg), float(ad), float(sd)
		num_iter.append(i)
		num_eval.append(e)
		abv.append(ab)
		stdbv.append(sb)
		agbv.append(ag)
		stdgbv.append(sg)
		advt.append(ad)
		stddvt.append(sd)
	return num_iter, num_eval, abv, stdbv, agbv, stdgbv, advt, stddvt

def get_last_result(project_name, experiment_name, stat_file_dirs, result_file_name):
	result_list = []
	for stat_file_dir in stat_file_dirs:
		n_iter, n_eval, abv, stbv, agv, stgv, adt, stdt = read_stat_file(project_name, experiment_name, stat_file_dir)
		result_list.append(stat_file_dir + ': ' + ' '.join([str(abv[-1]), str(stbv[-1])]) + '\n')
		
	result_list.sort(reverse = True)
	result_file = open(os.path.join(DATA_PATH, project_name, experiment_name, 'statFiles', result_file_name), 'w')
	for line in result_list:
		result_file.write(line)
	result_file.close()
	return 


class singular_visualize(object):
	""" a class used to implements common analysis on a single algorithm
	"""
	def __init__(self, project_name, experiment_name, stat_file_dir):
		self.project_name 		= project_name
		self.experiment_name 	= experiment_name
		self.stat_file_dir 		= stat_file_dir
		self.label 				= stat_file_dir[:stat_file_dir.find('_')]
		self.load_data()

	def load_data(self):
		n_iter, n_eval, abv, stbv, agv, stgv, adv, stdv = read_stat_file(self.project_name, self.experiment_name, self.stat_file_dir)
		self.n_iter, self.n_eval = np.array(n_iter), np.array(n_eval)
		self.abv, self.stbv = np.array(abv), np.array(stbv)
		self.agv, self.stgv = np.array(agv), np.array(stgv)
		self.adv, self.stdv = np.array(adv), np.array(stdv)

	def show_fitness(self):
		fitness_plot = Plot(os.path.join(DATA_PATH, self.project_name, self.experiment_name, 'visualizedData', self.stat_file_dir),\
										 [self.n_eval], [self.agv], [self.label], self.stat_file_dir+'_fitness.jpg')
		fitness_plot.line_plot()
		fitness_plot.set_xlabel('number of evaluations')
		fitness_plot.set_ylabel('fitness value')
		fitness_plot.set_legend()
		fitness_plot.save_plot()
		fitness_plot.close_plot()

	def show_diversity(self):
		diversity_plot = Plot(os.path.join(DATA_PATH, self.project_name, self.experiment_name, 'visualizedData', self.stat_file_dir),\
										   [self.n_eval], [self.adv], [self.label], self.stat_file_dir+'_diversity.jpg')
		diversity_plot.line_plot()
		diversity_plot.set_xlabel('number of evaluations')
		diversity_plot.set_ylabel('diversity')
		diversity_plot.set_legend()
		diversity_plot.save_plot()
		diversity_plot.close_plot()

	def show_fitness_diversity(self):
		fd_plot = Plot(os.path.join(DATA_PATH, self.project_name, self.experiment_name, 'visualizedData', self.stat_file_dir),\
										 np.vstack((self.n_eval, self.n_eval)), np.vstack((self.agv, self.adv)), \
										 [self.label, self.label], figName = self.stat_file_dir+'_fitness_diversity.jpg')
		fd_plot.line_plot()
		fd_plot.set_xlabel('number of evaluations')
		fd_plot.set_ylabel('fitness/diversity')
		fd_plot.set_legend()
		fd_plot.save_plot()
		fd_plot.close_plot()

class comparative_visualize(object):
	""" a class used to implements comparative analysis on more than one algorithm 
	"""
	def __init__(self, project_name, experiment_name, stat_file_dirs, func_name=''):
		self.project_name		= project_name
		self.experiment_name 	= experiment_name
		self.stat_file_dirs 	= stat_file_dirs
		self.labels 			= map(lambda x: x[:x.find('_')], stat_file_dirs)
		self.func_name 			= func_name
		self.result_file_dir 	= '_'.join(self.labels)
		self.load_data()

	def load_data(self):
		self.n_iter, self.n_eval, self.abv, self.stbv, self.agv, self.stgv, self.adv, self.stdv = [], [], [], [], [], [], [], []
		for stat_file_dir in self.stat_file_dirs:
			n_iter, n_eval, abv, stbv, agv, stgv, adv, stdv = read_stat_file(self.project_name, self.experiment_name, stat_file_dir)
			self.n_iter.append(np.array(n_iter))
			self.n_eval.append(np.array(n_eval))
			self.abv.append(np.array(abv))
			self.stbv.append(np.array(stbv))
			self.agv.append(np.array(agv))
			self.stgv.append(np.array(stgv))
			self.adv.append(np.array(adv))
			self.stdv.append(np.array(stdv))

	def show_fitness(self, semilogy = False):

		if self.func_name == '':
			plot_name = self.result_file_dir + '_fitness.jpg'
		else:
			plot_name = self.result_file_dir + '_%s_fitness.jpg' % self.func_name

		fitness_plot = Plot(os.path.join(DATA_PATH, self.project_name, self.experiment_name, 'visualizedData', self.result_file_dir),\
							self.n_eval, self.agv, self.labels, plot_name)

		if semilogy:
			fitness_plot.semilogy = True
			fitness_plot.figName = fitness_plot.figName[:fitness_plot.figName.rfind('.')] + '_logy.jpg'
		fitness_plot.line_plot()
		fitness_plot.set_xlabel('number of evaluations')
		fitness_plot.set_ylabel('fitness value')
		fitness_plot.set_legend()
		fitness_plot.save_plot()
		fitness_plot.close_plot()

	def show_diversity(self):
		if self.func_name == '':
			plot_name = self.result_file_dir + '_diversity.jpg'
		else:
			plot_name = self.result_file_dir + '_%s_diversity.jpg' % self.func_name

		diversity_plot = Plot(os.path.join(DATA_PATH, self.project_name, self.experiment_name, 'visualizedData', self.result_file_dir),\
							self.n_eval, self.adv, self.labels, plot_name)
		diversity_plot.line_plot()
		diversity_plot.set_xlabel('number of evaluations')
		diversity_plot.set_ylabel('fitness value')
		diversity_plot.set_legend()
		diversity_plot.save_plot()
		diversity_plot.close_plot()
