
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция для вычисления значения фитнес-функции на основе координат x и y
def fitness_function(x, y):
    return np.cos(x) * np.cos(y) * np.exp(-x**2 - y**2)

# Функция для генерации начальной популяции
def generate_population(size):
    population = []
    for _ in range(size):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        chromosome = (x, y)
        population.append(chromosome)
    return population

# Функция для выполнения селекции особей на основе значения фитнес-функции
def selection(population):
    sorted_population = sorted(population, key=lambda chromosome: fitness_function(*chromosome), reverse=True)
    return sorted_population[0]

# Функция для выполнения скрещивания особей
def crossover(parent1, parent2):
    x1, y1 = parent1
    x2, y2 = parent2
    child_x = (x1 + x2) / 2
    child_y = (y1 + y2) / 2
    return child_x, child_y

# Функция для выполнения мутации особи
def mutation(chromosome, mutation_probability):
    x, y = chromosome
    if random.random() < mutation_probability:
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
    return x, y

# Функция для визуализации поверхности фитнес-функции
def visualize_surface():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, y)
    z = fitness_function(x, y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    plt.show()

# Функция для визуализации результатов расчета

def visualize(population, generation):
    x_values = []
    y_values = []
    z_values = []
    labels = []
    coordinates_dict = {}

    for i, chromosome in enumerate(population):
        x, y = chromosome
        coordinates = (x, y)
        if coordinates in coordinates_dict:
            coordinates_dict[coordinates].append(i + 1)
        else:
            coordinates_dict[coordinates] = [i + 1]

        x_values.append(x)
        y_values.append(y)
        z_values.append(fitness_function(x, y))
        labels.append(str(i+1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x_values, y_values, z_values, c=z_values, cmap='viridis')

    for coordinates, label_indices in coordinates_dict.items():
        x, y = coordinates
        z = fitness_function(x, y)
        if len(label_indices) > 1:  # Если есть более чем один индекс для этой координаты
            label = ','.join(map(str, label_indices))
        else:
            label = str(label_indices[0])  # В противном случае, используйте единственный индекс
        ax.text(x, y, z, label, color='black')


    ax.set_title(f'Generation {generation}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')

    fig.colorbar(img)
    plt.show()
# Значение N для критерия останова
N = 80001

# Генерация начальной популяции
population = generate_population(4)

# Визуализация поверхности фитнес-функции
visualize_surface()

# Выполнение простого генетического алгоритма
fitness_values = [fitness_function(*chromosome) for chromosome in population]

for i in range(N):
    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 80000]:
        max_fitness = max(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        print("\nGeneration {}: ".format(i))
        print("{:<12} {:<12} {:<12} {:<12}".format('Chromosome', 'x', 'y', 'fitness'))
        for j, chromosome in enumerate(population):
            x, y = chromosome
            fitness = fitness_values[j]
            print("{:<12} {:<12.6f} {:<12.6f} {:<12.6f}".format(j+1, x, y, fitness))
        print("Max fitness in generation {}: {:.6f}".format(i, max_fitness))
        print("Average fitness in generation {}: {:.6f}".format(i, average_fitness))

        visualize(population, i)

    best_chromosome = selection(population)
 
    children = [crossover(best_chromosome, chromosome) for chromosome in population[1:]]
    
    mutation_probability = 0.1
    population = [best_chromosome] + [mutation(child, mutation_probability) for child in children]
    fitness_values = [fitness_function(*chromosome) for chromosome in population]

# Получение оптимального значения функции и соответствующих координат
best_chromosome = selection(population)
best_fitness = fitness_function(*best_chromosome)
print("\nOptimal solution: {:.6f}, {:.6f}".format(best_chromosome[0], best_chromosome[1]))
print("Fitness: {:.6f}".format(best_fitness))

