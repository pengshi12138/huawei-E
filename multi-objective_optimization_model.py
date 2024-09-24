import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms

# 自定义目标函数，根据拥塞参数判断是否开启应急通道
def evaluate(individual):
    # 个体包含的参数
    flow_sum = individual[0]        # 车流量
    density = individual[1]         # 车密度
    residence_time = individual[2]  # 滞留时间
    lanes_sum = individual[3]       # 道路数量

    # 检查约束条件
    if flow_sum < 0 or flow_sum > 1:
        return (-1,)  # 给予惩罚，表示此个体无效
    if density < 0 or density > 1:
        return (-1,)  # 给予惩罚，表示此个体无效

    # 计算拥塞指数
    congestion_index = (flow_sum * 0.33 + density * 0.33 + residence_time * 0.33) / lanes_sum

    # 返回目标，最小化开启应急通道的数量
    if congestion_index > 0.25 and flow_sum > 0.6 and density > 0.75 and residence_time > 0.8:
        return (1,)
    else:
        return (0,)

# 数据加载
def load_data():
    # 读取CSV文件
    data1 = pd.read_excel('data/all_data.xlsx')

    # 合并数据
    data = pd.concat([data1])

    return data

# 设置遗传算法参数
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化目标
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 生成符合范围的个体
toolbox.register("flow_sum", np.clip, np.random.uniform(0, 1), 0, 1)
toolbox.register("density", np.clip, np.random.uniform(0, 1), 0, 1)
toolbox.register("residence_time", np.clip, np.random.uniform(0, 1), 0, 1)
toolbox.register("lanes_sum", np.random.randint, 2, 3)  # 道路数量

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.flow_sum,
                  toolbox.density, toolbox.residence_time, toolbox.lanes_sum), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)  # 使用锦标赛选择

# 边界修正函数
def check_bounds(individual):
    # 检查并修正超出范围的值
    individual[0] = np.clip(individual[0], 0, 1)  # 车流量
    individual[1] = np.clip(individual[1], 0, 1)  # 车密度
    individual[2] = np.clip(individual[2], 0, 1)  # 滞留时间
    individual[3] = int(np.clip(individual[3], 2, 3))  # 道路数量
    return individual

# 主程序
if __name__ == "__main__":
    data = load_data()

    # 生成初始种群
    population = toolbox.population(n=200)  # 增加种群规模

    # 运行遗传算法
    NGEN = 300  # 增加迭代次数
    CXPB = 0.7  # 增加交叉概率
    MUTPB = 0.2  # 保持变异概率

    for gen in range(NGEN):
        # 选择
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < CXPB:
                toolbox.mate(child1, child2)
                child1 = check_bounds(child1)
                child2 = check_bounds(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < MUTPB:
                toolbox.mutate(mutant)
                mutant = check_bounds(mutant)
                del mutant.fitness.values

        # 评估所有个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新种群
        population[:] = offspring

    # 输出最优解
    fits = [ind.fitness.values[0] for ind in population]
    best_idx = np.argmin(fits)
    print("最优个体:", population[best_idx])