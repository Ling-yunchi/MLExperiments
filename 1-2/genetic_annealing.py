import numpy as np
import random


# 目标函数
def fitness(x):
    return x * np.sin(10 * np.pi * x) + 1.0


# 遗传算法主程序
def genetic_algorithm(bounds, population_size, num_generations, mutation_rate):
    # 初始化种群
    def init_population(size, bounds):
        return [random.uniform(bounds[0], bounds[1]) for _ in range(size)]

    # 选择操作
    def select(population, fitnesses, num_parents):
        parents = []
        for _ in range(num_parents):
            max_fitness_idx = np.where(fitnesses == np.max(fitnesses))
            parents.append(population[max_fitness_idx[0][0]])
            fitnesses[max_fitness_idx[0][0]] = -999999
        return parents

    # 交叉操作
    def crossover(parents, offspring_size):
        offspring = []
        for _ in range(offspring_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    # 变异操作
    def mutate(offspring, mutation_rate, bounds):
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] += random.uniform(-0.1, 0.1)
                # 确保变异后的个体仍在范围内
                offspring[i] = min(max(offspring[i], bounds[0]), bounds[1])
        return offspring

    # 初始种群
    population = init_population(population_size, bounds)

    for generation in range(num_generations):
        # 计算适应度
        fitnesses = [fitness(x) for x in population]
        # 选择
        parents = select(population, np.array(fitnesses), population_size // 2)
        # 交叉
        offspring = crossover(parents, population_size - len(parents))
        # 变异
        offspring = mutate(offspring, mutation_rate, bounds)
        # 更新种群
        population = parents + offspring

    # 找到最优解
    best_solution = max(population, key=fitness)
    return best_solution, fitness(best_solution)


# 模拟退火算法
def simulated_annealing(max_iterations=1000, initial_temp=100, cooling_rate=0.99):
    current_x = random.uniform(-1, 2)
    current_fitness = fitness(current_x)
    best_x = current_x
    best_fitness = current_fitness

    temperature = initial_temp
    for i in range(max_iterations):
        # 在邻域中随机选择一个新解
        new_x = current_x + random.uniform(-0.1, 0.1)
        new_x = np.clip(new_x, -1, 2)
        new_fitness = fitness(new_x)

        # 如果新解更优，则接受新解
        if new_fitness > current_fitness:
            current_x = new_x
            current_fitness = new_fitness
            if new_fitness > best_fitness:
                best_x = new_x
                best_fitness = new_fitness
        else:
            # 否则以一定概率接受新解
            acceptance_prob = np.exp((new_fitness - current_fitness) / temperature)
            if random.random() < acceptance_prob:
                current_x = new_x
                current_fitness = new_fitness

        # 降温
        temperature *= cooling_rate

    return best_x, best_fitness


# 主程序
bounds = (-1, 2)
population_size = 50
num_generations = 1000
mutation_rate = 0.01

# 运行遗传算法
best_x_ga, best_fitness_ga = genetic_algorithm(bounds, population_size, num_generations, mutation_rate)
print("遗传算法：最佳x =", best_x_ga, "最大值 =", best_fitness_ga)

# 运行模拟退火算法
best_x_sa, best_fitness_sa = simulated_annealing()
print("模拟退火算法：最佳x =", best_x_sa, "最大值 =", best_fitness_sa)
