import random
import time

import numpy as np
from matplotlib import pyplot as plt


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
                offspring[i] = random.uniform(bounds[0], bounds[1])
        return offspring

    # 初始种群
    population = init_population(population_size, bounds)
    best_fitness_history = []  # 存储每一代的最佳适应度值
    best_individual_history = []  # 存储每一代最佳适应度值对应的个体

    for generation in range(num_generations):
        # 计算适应度
        fitnesses = [fitness(x) for x in population]
        best_fitness_history.append(max(fitnesses))  # 记录这一代的最佳适应度值
        best_individual_history.append(
            population[np.argmax(fitnesses)]
        )  # 记录这一代最佳适应度值对应的个体
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
    return (
        best_solution,
        fitness(best_solution),
        best_fitness_history,
        best_individual_history,
    )


# 模拟退火算法
def simulated_annealing(bounds, max_iter, initial_temp, cooling_rate):
    current_x = random.uniform(bounds[0], bounds[1])
    current_fitness = fitness(current_x)
    best_x = current_x
    best_fitness = current_fitness
    fitness_history = []  # 存储每次迭代后当前解的适应度值
    best_fitness_history = []  # 存储每次迭代后最佳解的适应度值
    current_x_history = []  # 存储每次迭代后当前解的数值
    temperature_history = []  # 存储每次迭代后的温度

    temperature = initial_temp
    for i in range(max_iter):
        # 在邻域中随机选择一个新解
        new_x = current_x + random.uniform(-1, 1) * temperature
        new_x = np.clip(new_x, bounds[0], bounds[1])
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

        fitness_history.append(current_fitness)  # 记录当前解的适应度值
        best_fitness_history.append(best_fitness)  # 记录最佳解的适应度值
        current_x_history.append(current_x)  # 记录当前解的数值
        temperature_history.append(temperature)  # 记录当前温度

        # 降温
        temperature *= 1 - cooling_rate

    return (
        best_x,
        best_fitness,
        fitness_history,
        best_fitness_history,
        current_x_history,
        temperature_history,
    )


# 主程序
bounds = (-1, 2)
population_size = 50
num_generations = 1000
mutation_rate = 0.2

# 运行遗传算法
ti = time.time()
best_x_ga, best_fitness_ga, ga_fitness_history, ga_individual_history = (
    genetic_algorithm(bounds, population_size, num_generations, mutation_rate)
)
print(f"运行时间：{time.time() - ti:.5f}")
print("遗传算法：最佳x =", best_x_ga, "最大值 =", best_fitness_ga)

max_iter = 1000
initial_temp = 100
cooling_rate = 0.01

# 运行模拟退火算法
ti = time.time()
(
    best_x_sa,
    best_fitness_sa,
    sa_fitness_history,
    sa_best_fitness_history,
    sa_current_x_history,
    sa_temperature_history,
) = simulated_annealing(bounds, max_iter, initial_temp, cooling_rate)
print(f"运行时间：{time.time() - ti:.5f}")
print("模拟退火算法：最佳x =", best_x_sa, "最大值 =", best_fitness_sa)


def plot_genetic_algorithm(ga_fitness_history, ga_individual_history):
    # 创建一张包含两个子图的图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1：每代的最佳适应度
    ax1.plot(ga_fitness_history, label="Best Fitness", color="blue")
    ax1.set_title("Genetic Algorithm - Best Fitness Over Generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.grid(True)
    ax1.legend()

    # 图2：每代的最佳个体（x 值）
    ax2.plot(ga_individual_history, label="Best Individual (x)", color="green")
    ax2.set_title("Genetic Algorithm - Best Individual (x) Over Generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Individual (x)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# 模拟退火算法可视化
def plot_simulated_annealing(
    sa_fitness_history,
    sa_best_fitness_history,
    sa_current_x_history,
    sa_temperature_history,
):
    # 创建一张包含两个子图的图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1：当前最优适应度 vs 历史最优适应度
    ax1.plot(sa_fitness_history, label="Current Best Fitness", color="red")
    ax1.plot(
        sa_best_fitness_history,
        label="Historical Best Fitness",
        color="blue",
        linestyle="--",
    )
    ax1.set_title("Simulated Annealing - Fitness Over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.grid(True)
    ax1.legend()

    # 图2：温度变化和当前 x 值变化
    ax2.plot(sa_current_x_history, label="Current x", color="purple")
    ax2.set_title("Simulated Annealing - Current x Over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Current x")
    ax2.grid(True)

    ax2_twinx = ax2.twinx()
    ax2_twinx.set_ylabel("Temperature", color="orange")
    ax2_twinx.plot(sa_temperature_history, label="Temperature", color="orange")
    ax2_twinx.tick_params(axis="y", labelcolor="orange")

    ax2.legend()

    plt.tight_layout()
    plt.show()


# 可视化
plot_genetic_algorithm(ga_fitness_history, ga_individual_history)
plot_simulated_annealing(
    sa_fitness_history,
    sa_best_fitness_history,
    sa_current_x_history,
    sa_temperature_history,
)
