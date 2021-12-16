from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import array
import random
import numpy as np
from deap import base
import math,time
from deap import creator
from deap import tools
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import geatpy as ea
from grid_idea2 import grid_dominance

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


def fit_train(x1, train_data):
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x= random.choice(x)
    x = 1 * (x >= 0.6)
    if np.count_nonzero(x) == 0:
        f1 = 1###error_rate
        f2 = 1
    else:
     x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
     value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
     value_position = np.insert(value_position, 0, 0)  # insert the column of label
     tr = train_data[:, value_position]
     clf = KNeighborsClassifier(n_neighbors = 5)
     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 10)
     f1 = np.mean(1 - scores)
     f2 = (len(value_position)-1)/(train_data.shape[1] - 1)
     # f2 = len(value_position) - 1
    return f1, f2

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   # shape[0]表示行数
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = squaredDiff.sum(axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sorted_ClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ClassCount[0][0]


def evaluate_test_data(x, train_data, test_data):
    x = 1 * (x >= 0.6)
    x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
    value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
    value_position = np.insert(value_position, 0, 0)  # insert the column of label
    te = test_data[:, value_position]#####testing data including label in the first colume
    tr = train_data[:, value_position]#####training data including label in the first colume too
    wrong = 0
    for i12 in range(len(te)):
        testX = te[i12,1:]
        dataSet = tr[:,1:]
        labels = tr[:,0]
        outputLabel = kNNClassify(testX, dataSet, labels, 5)
        if outputLabel != te[i12,0]:
            wrong = wrong + 1
    f1 = wrong/len(te)
    f2 = (len(value_position) - 1) / (test_data.shape[1] - 1)
    return f1, f2

def first_level_nondominant(x1):
    x = np.array([ind.fitness.values for ind in x1])
    [levels, criLevel] = ea.indicator.ndsortDED(x, 1)
    x = 1 * (levels == 1.0)
    x = "".join(map(str, x))
    index = np.array(list(find_all(x, '1')))
    PF = [x1[i_1] for i_1 in index]
    return PF

def first_nondominated(pop):
    PF = np.array([ind.fitness.values for ind in pop])
    [levels1, criLevel1] = ea.indicator.ndsortDED(PF, 1)
    x1 = 1 * (levels1 == 1.0)
    x1 = "".join(map(str, x1))
    index1 = np.array(list(find_all(x1, '1')))
    pop_non = [pop[i] for i in index1]
    return pop_non


def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result


def more_confidence(EXA, index_of_objectives):
    a = 0.6
    cr = np.zeros((len(index_of_objectives),1))
    for i in range(len(index_of_objectives)):###the number of indexes
        temp = 0
        object = EXA[index_of_objectives[i]]
        for ii in range(len(object)):###the number of features
           b = object[ii]
           if b > a:  con = (b - a) / (1 - a)
           else:      con = (a - b) / (a)
           temp = con + temp
        cr[i,0] = temp
    sorting = np.argsort(-cr[:,0])####sorting from maximum to minimum
    index_one = index_of_objectives[sorting[0]]
    return index_one

def delete_duplicate(EXA):####list
    EXA1 = []
    EXA_array = np.array(EXA)
    all_index = []
    for i0 in range(EXA_array.shape[0]):
       x = 1 * (EXA_array[i0,:] >= 0.6)
       x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
       all_index.append(x)##store all individuals who have changed to 0 or 1
    single_index = set(all_index)####find the unique combination
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(single_index)):
       index_of_objectives = findindex(all_index, single_index[i1])##find the index of each unique combination
       if len(index_of_objectives) == 1:####some combination have more than one solutions.
          for i2 in range(len(index_of_objectives)):
             EXA1.append(EXA[index_of_objectives[i2]])
       else:
           index_one = more_confidence(EXA, index_of_objectives)
           EXA1.append(EXA[index_one])
    return EXA1

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))####minimise two objectives
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def main_loop(seed,x_train):
    random.seed(seed)
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    NDIM = x_train.shape[1] - 1
    NGEN = 100###the number of generation
    if NDIM < 300:
        MU = 4*math.floor(NDIM/4)  ####the number of particle
    else:
        MU = 300  #####bound to 300
    CXPB = 0.9
    Max_FES = MU * NGEN
    fit_num = 0
    div = 15
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)  #####dertemine the way of randomly generation and gunrantuu the range
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)  ###fitness
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ##particles
    toolbox.register("evaluate", fit_train, train_data= x_train)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
    pop = toolbox.population(n=MU)
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)#####toolbox.evaluate = fit_train
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    fit_num = fit_num + MU
    unique_number = []
    pop_surrogate = delete_duplicate(pop)
    unique_number.append(len(pop_surrogate))
    for gen in range(1, NGEN):
        offspring = toolbox.clone(pop)
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):###it starts from the zeros and first individual. the interval is 2.
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)# Apply mutation on the offspring
            toolbox.mutate(ind2)# Apply mutation on the offspring
            del ind1.fitness.values, ind2.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        pop_surrogate.extend(delete_duplicate(offspring))
        pop_surrogate = delete_duplicate(pop_surrogate)
        unique_number.append(len(pop_surrogate))
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        fit_num = fit_num + MU
        pop_mi = pop + offspring
        pop1 = delete_duplicate(pop_mi)
        if len(pop1) == MU:
            pop = pop1
        elif len(pop1) > MU:
            pop = grid_dominance(pop1,div,MU)
        if fit_num > Max_FES:
            break
    return pop,unique_number
