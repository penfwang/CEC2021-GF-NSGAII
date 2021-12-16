#some concepts used from the following paper
###Yang S, Li M, Liu X, et al. A grid-based evolutionary algorithm for many-objective optimization[J].
# IEEE Transactions on Evolutionary Computation, 2013, 17(5): 721-736.
from __future__ import division
import numpy as np
import math
import random
import geatpy as ea

def uniform(low, up, size=None):####generate a matrix of the range of variables
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result



def calcu_GR(grid_coor):#grid_coor all grid coordinate
    GR = [0.0]*len(grid_coor)
    for j in range(len(grid_coor)):
        GR[j] = grid_coor[j][0] + grid_coor[j][1]
    return GR





def calcu_GCD(G,index_pop):#G all the grid axis, index_pop, the insex of the solutions with the same GR
    GD = np.zeros((len(index_pop),len(G)))
    GCD = [0.0] * len(index_pop)
    ####the first step is to find the neighbors of all the solutions
    for i in range(len(index_pop)):
        for j in range(len(G)):
            GD[i,j] = 2 - abs(G[index_pop[i]][0] - G[j][0]) - abs(G[index_pop[i]][1] - G[j][1])
        s1 = [x for x in GD[i] if x > 0]
        GCD[i] = sum(s1)
    ######## the distance between GD is smaller than M(2) which can be definited as neighbors
    return GCD



def multiple_combine(GR,GCD,f,num):
    s1 = np.argsort(GR)
    GR_sort = sorted(GR)
    if GR_sort[num - 1] == GR_sort[num]:####
         s2 = np.argsort(GCD)
         ina = f[s2[0:num]]
         return ina
    else:
        ina = f[s1[0:num]]
        return ina


def infor_smaller_selection_point(x,x_sort,num,new_x):
    selection_index = []
    in1 = np.argwhere(x < x_sort[num - 1])
    if len(in1) == 0:
        in2 = np.argwhere(x == x_sort[num - 1])
        x_new = [new_x[in2[j][0]] for j in range(len(in2))]
        in3 = np.argsort(x_new)
        selection_index = [in2[in3[j]] for j in range(len(in3[0:num]))]#[array([0])]
        return selection_index
    else:
        selection_index.extend(in1)
        new_num = num - len(in1)
        ind = np.argwhere(x == x_sort[num - 1])  ##find the solutions in the third front who have the same GR
        x_new = [new_x[ind[j][0]] for j in range(len(ind))]
        ind1 = np.argsort(x_new)
        s2 = [ind[ind1[j]] for j in range(len(ind1[0:new_num]))]
        selection_index.extend(s2)#[array([0]),array([1])]
        return selection_index



def multiple_combine_new(GR,GCD,f,num):###consider the selection point duplication
    s1 = np.argsort(GR)
    GR_sort = sorted(GR)
    if GR_sort[num - 1] == GR_sort[num]:####
         se_index = infor_smaller_selection_point(GR,GR_sort,num,GCD)
         ina = [f[se_index[j][0]] for j in range(len(se_index))]
         return ina
    else:
        ina = f[s1[0:num]]
        return ina



def multiple_combine3(GR,GCD,GPCD,f,num):
    s1 = np.argsort(GR)
    GR_sort = sorted(GR)
    GCD_sort = sorted(GCD)
    if GR_sort[num - 1] == GR_sort[num]:####GR duplication
        if GCD_sort[num - 1] == GCD_sort[num]:
            s3 = np.argsort(GPCD)
            ina = f[s3[0:num]]
            return ina
        else:
             s2 = np.argsort(GCD)
             ina = f[s2[0:num]]
             return ina
    else:
        ina = f[s1[0:num]]
        return ina



def grid_dominance(pop, div, num):
    new_pop = []
    pop_fit = np.array([ind.fitness.values for ind in pop])
    g = np.zeros((len(pop), 2))
    for j in range(len(pop_fit[0])):
       lb = min(pop_fit[:, j]) - (max(pop_fit[:, j]) - min(pop_fit[:, j])) / (2 * div)
       ub = max(pop_fit[:, j]) + (max(pop_fit[:, j]) - min(pop_fit[:, j])) / (2 * div)
       d = (ub - lb) / div
       if d == 0:
           new_pop = random.sample(pop,num)
           return new_pop
       for i in range(len(pop)):
         g[i,j] = math.floor((pop_fit[i][j]-lb)/d)
    whole_GR = calcu_GR(g)
    [levels1, criLevel1] = ea.indicator.ndsortDED(g)
    set_le = list(set(levels1))  ####grid dominance sorting
    temp = {}
    temp1 = 0
    for f in range(len(set_le)):
         x1 = 1 * (levels1 == set_le[f])
         x1 = "".join(map(str, x1))
         index = np.array(list(find_all(x1, '1')))
         temp[f] = index
         temp1 += len(index)####the front level achieve the pop size
         if temp1 == num:###don't need the three critera
        ###Based on the non-dominated sorting, the number of solutions in the first several fronts are equal to num.
             for le in temp:
                new_pop.extend(temp[le])###final output
             output = [pop[ii] for ii in new_pop]
             return output
         elif temp1 > num:
            if len(temp) > 1:
               for i4 in range(len(temp) - 1):
                    new_pop.extend(temp[i4])  ###obtain the first two fronts
               save_f_num = num - (temp1 - len(temp[f])) ##in the set_le[f] front, the num need to be saved
               temp_g = [g[ii] for ii in temp[f]]##grid coordinate of the solutions in the temp[f] front
               GR = calcu_GR(temp_g)##GR values of the solutions in the temp[f] front
               GCD = calcu_GCD(g,temp[f])
               indexa = multiple_combine_new(GR,GCD,temp[f], save_f_num)
               for i in range(save_f_num):
                   new_pop.extend([indexa[i]])
               output = [pop[ii] for ii in new_pop]
               return output
            elif len(temp) == 1:
                # print('the first front includes more than num solutions')###need more cares
                temp_1 = [g[ii] for ii in temp[f]]
                GR = calcu_GR(temp_1)
                GCD = calcu_GCD(g, temp[f])
                indexa = multiple_combine_new(GR, GCD, temp[f], num)
                for i in range(num):
                    new_pop.extend([indexa[i]])
                output = [pop[ii] for ii in new_pop]
                return output