from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import random
import numpy as np
import math,time
import geatpy as ea
from our_operators import main_loop,evaluate_test_data

##### Please cite our paper,
## A Grid-dominance based Multi-objective Algorithm for Feature Selection in Classification
# @inproceedings{wang2021grid,
#   title={A Grid-dominance based Multi-objective Algorithm for Feature Selection in Classification},
#   author={Wang, Peng and Xue, Bing and Zhang, Mengjie and Liang, Jing},
#   booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)},
#   pages={2053--2060},
#   year={2021},
#   organization={IEEE}
# }

if __name__ == "__main__":
    tt = ['dataSet_wine']
    seed=[16]
    dataset_name = tt[0]
    folder1 = '/vol/grid-solar/sgeusers/wangpeng/multi-result/split_73' + '/' + 'train' + str(dataset_name) + ".npy"
    folder2 = '/vol/grid-solar/sgeusers/wangpeng/multi-result/split_73' + '/' + 'test' + str(dataset_name) + ".npy"
    x_train = np.load(folder1)
    x_test = np.load(folder2)
    running_30 = []
    hyp_30_training = []
    hyp_30_testing = []
    refrence_point = np.ones((1, 2))
    for i in range(len(seed)):
       start = time.time()
       random.seed(seed[i])
       pop,unique_number = main_loop(seed[i],x_train)
       end = time.time()
       running_time = end - start
       running_30.append(running_time)
       front_training = np.array([ind.fitness.values for ind in pop])
       hyp_training = ea.indicator.HV(front_training, refrence_point)
       hyp_30_training.append(hyp_training)
       EXA_array = np.array(pop)
       EXA_01 = 1 * (EXA_array >= 0.6)
       front_testing = np.ones((EXA_array.shape[0], 2))
       for i in range(EXA_array.shape[0]):
           front_testing[i, :] = evaluate_test_data(EXA_array[i, :], x_train, x_test)
       hyp_testing = ea.indicator.HV(front_testing, refrence_point)
       hyp_30_testing.append(hyp_testing)
    print(hyp_30_testing)
    print('End')


