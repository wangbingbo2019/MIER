import datetime
import math
import numpy as np
import pandas as pd
import scipy.io as scio
import random
from matplotlib import pyplot as plt
from numpy.linalg import *
from deap import base, creator, algorithms
from deap import tools
from celluloid import Camera
# 单目标最小化问题，使得能量最小
from scipy.stats import bernoulli
from utils.utils import *

creator.create("FitnessMin", base.Fitness, weights=(-1,-1))
creator.create("Individual", list, fitness=creator.FitnessMin)

# INF_SIZE个体长度
# IND_SIZE = 994
IND_SIZE = 7555

toolbox = base.Toolbox()
toolbox.cxProb = 0.7
# 顺序打乱变异
# toolbox.mutateProb = 0.5
# 位反转突变
toolbox.mutateProb = 0.8
toolbox.popSize = 50

def create_binary(num):

    # ***以0.05的概率赋1，其余赋0***
    # ind = []
    # for j in range(IND_SIZE):
    #     # if random.random()<(num+1)/toolbox.popSize:
    #     if random.random() < 0.05:
    #         ind.append(1)
    #     else:
    #         ind.append(0)
    rd_index = random.randint(0, IND_SIZE-1)
    ind = np.zeros(IND_SIZE)
    ind[rd_index] = 1

    return ind


toolbox.register('binary', bernoulli.rvs, 0.5) #注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = IND_SIZE) #用tools.initRepeat生成长度为GENE_LENGTH的Individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# 交叉方法
toolbox.register("mate", tools.cxTwoPoint)
# 变异方法
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

# toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.002)

# 选择方法
toolbox.register("select", tools.selTournament, tournsize=3)

energy_dict = dict()
def evaluate(individual):
    temp_list=list()
    save_list = list()
    pos = 0
    num = 0
    # ~ try:
    for item in individual:
        if item == 0:
            temp_list.append(pos)
        else:
            num += 1
            save_list.append(pos)
        pos += 1
    if tuple(save_list) in energy_dict.keys():
        # print('重复使用',save_list)
        E = energy_dict[tuple(save_list)]
    else:
        # add_A21 = np.delete(re_A21, temp_list, 1)
        add_A21 = re_A21[:,save_list]
        A21_copy = np.hstack((o_A21, add_A21))
        # print(A21_copy)
        E = e_expect_2(A21_copy)
        energy_dict[tuple(save_list)] = E
    print(E,num,save_list)
    if float(E)< 0 or np.isnan(E):
        return 9e+20,num
    elif num == 0:
        return 9e+20,99999
    else:
        return float(E),num

# 评估方法
toolbox.register("evaluate", evaluate)

def main():

    #初始化种群大小
    pop = toolbox.population(n=toolbox.popSize)
    for i in range(toolbox.popSize):
        pop[i] = creator.Individual(create_binary(i))

    # pop = create_binary()
    # print(type(pop[0]))


    #NGEN 迭代次数
    NGEN = 500

    plt.figure(figsize=(8 / 2.54, 8 / 2.54))
    fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)
    # 将每个个体的适应度设置为pareto前沿的次序
    for idx, front in enumerate(fronts):
        for ind in front:
            print(idx, ind)
            ind.fitness.values = (idx + 1),

    offspring = toolbox.select(pop, len(pop))
    offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)


    print("  Evaluated %i individuals" % len(pop))
    print("-- Iterative %i times --" % NGEN)
    for g in range(NGEN):
        print("-- Generation %i --" % g,datetime.datetime.now())
        combinedPop = pop + offspring  # 合并父代与子代
        # 评价族群
        fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
        for ind, fit in zip(combinedPop, fitnesses):
            ind.fitness.values = fit

        # 快速非支配排序
        fronts = tools.emo.sortNondominated(combinedPop, k=toolbox.popSize, first_front_only=False)
        # 拥挤距离计算
        for front in fronts:
            tools.emo.assignCrowdingDist(front)
            # 环境选择 -- 精英保留
            pop = []
            for front in fronts:
                pop += front
            pop = toolbox.clone(pop)
            pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')
        # 创建子代
        offspring = toolbox.select(pop, toolbox.popSize)
        offspring = toolbox.clone(offspring)
        if g == 2:
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.8)
        offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)


        if g % 1 ==0 :

            # 每次迭代保存图片
            s_front = tools.emo.sortNondominated(pop, len(pop))[0]
            df_front = pd.DataFrame()
            # 每次绘图清除前一次画布中的点
            plt.clf()
            # 指定绘图的坐标轴范围
            # plt.xlim(0, 50)
            # plt.ylim(ymin=100,ymax=700)
            solution_set = set()
            # 遍历帕累托前沿中的解
            for s_ind in s_front:
                solution_set.add(s_ind.fitness)
            for s_ind in s_front:
                if s_ind.fitness in solution_set:
                    # 打印解（能量，节点数） [节点序号]
                    result = print_individual(s_ind)
                    # 保证存入的解都不重复
                    solution_set.remove(s_ind.fitness)
                    # 新加入的解
                    new_df = pd.DataFrame({'gen': g,
                                          'energy': s_ind.fitness.values[0],
                                           'num': s_ind.fitness.values[1],
                                           'node': ' '.join(list(map(str, result)))
                                           }, index= [len(df_front)])
                    # # 节点序号
                    # length =0
                    # node = []
                    # for node_num in range(IND_SIZE):
                    #     if s_ind[node_num]!=0:
                    #         new_df['node'] = node_num
                    #         length+=1
                    df_front = df_front.append(new_df)
                    # 绘图
                    plt.plot(s_ind.fitness.values[1], s_ind.fitness.values[0], 'r.', ms=2)

            df_front.to_csv('结果/test1-规模%i.csv' %toolbox.popSize,mode=('w' if g==0 else 'a'),  header=(True if g==0 else False))
            # plt.title('Pareto optimal front derived with NSGA-II', fontsize=12)
            plt.xlabel('num', fontsize=11, fontproperties='Times New Roman')
            plt.ylabel('energy', fontsize=11, fontproperties='Times New Roman')
            plt.tight_layout()
            plt.savefig('结果/Pareto_optimal_front_derived_with_NSGA-II-%i.png' %g)

    print("-- End of (successful) evolution --",datetime.datetime.now())
    return  pop


if __name__ == "__main__":

    df = pd.read_csv('D:\mqd\python_practice\MOEA\case1癌症基因网络\data\BLCA\\test1\o_A21.csv', index_col=0)
    o_A21 = df.values
    df = pd.read_csv('D:\mqd\python_practice\MOEA\case1癌症基因网络\data\BLCA\\test1\\re_A21.csv', index_col=0)
    re_A21 = df.values

    pop = main()


