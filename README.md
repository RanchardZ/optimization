# optimization
This package is used for numeric optimization of single objective problem
Author:       Ranchard Zheng
Email:        ranchardzheng@ruc.edu.cn
Affiliation:  MMC Lab of Renmin University of China

After cloning this package, you need to modify the CODE_PATH and DATA_PATH in constant.py

The basic usage would be:

from bee import ABC
spe = ABC(50, 30, 'Rastrigin', max_evaluations = 300000, observe_points = range(10000, 310000, 10000))
spe.evolveSpecies()

fist argument refers to the population size, second one refers to the number of problem dimensions
other arguments are self-explained.



if you specify the keyword argument (archive_switch = True, projectName = some_name, experimentName = another_name) 
in the initialization like this

spe = ABC(50, 30, 'Rastrigin', max_evaluations = 300000, observe_points = range(10000, 310000, 10000),\
           archive_switch = True, projectName = 'test', experimentName = 'testABC')
           
the data collected at every iteration will be stored to the DATA_PATH/projectName/experimentName
