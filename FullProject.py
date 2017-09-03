import numpy as np
from itertools import combinations
import math
import random
import matplotlib.pyplot as plt
import heapq
import itertools

ranges = {'Sphere': (-5.12, 5.12), 'Rosenbrock': (-2.048, 2.048), 'Rastrigin': (-5.12, 5.12), 'Schwefel': (-512.03, 511.97), 'Ackley': (-30, 30), 'Griewangk': (-600, 600)}
#key = 'Schwefel'

#For Genetic Algorithm
generation_size = 100
num_of_generations = 1000

#For mutation
mutation_probability = 1.0 / 30.0

#For selection
elite_percent = .05
elite = int(elite_percent * generation_size)
tournament_size = 5

#For crossover
crossover_methods = ('one-point', 'two-point', 'uniform', 'arithmetic')
#crossover_method = crossover_methods[0]
mixing_ratio = 0.5    #Used in uniform crossover, chance of gene swap

def fitness(values, func):
  if func == 'Sphere':
    return sum([val**2 for val in values])
  elif func == 'Rosenbrock':
    return sum([100*(values[i+1] - values[i]**2)**2 + (values[i] - 1)**2 for i in xrange(len(values)-1)])
  elif func == 'Rastrigin':
    return 10 * len(values) + sum([values[i]**2 - 10 * math.cos(2 * math.pi * values[i]) for i in xrange(len(values))])
  elif func == 'Schwefel':
    return 418.982887 * len(values) + sum([values[i] * math.sin(math.sqrt(abs(values[i]))) for i in xrange(len(values))])
  elif func == 'Ackley':
    sum1 = sum([values[i]**2 for i in xrange(len(values))])
    sum2 = sum([math.cos(2 * math.pi * values[i]) for i in xrange(len(values))])
    return 20.0 + math.e - 20.0 * math.exp(-0.2 * math.sqrt((1.0/float(len(values))) * float(sum1))) - math.exp((1.0/float(len(values))) * float(sum2))
  elif func == 'Griewangk':
    product = math.cos(float(values[0])/math.sqrt(1))
    for i in xrange(1, len(values)):
      product *= math.cos(float(values[i])/math.sqrt(i+1))
    sum1 = sum([values[i]**2 / 4000 for i in xrange(len(values))])
    return 1 + sum1 - product
  else:
    assert False, str(func) + ' is not a benchmark name.'

def mutation(inputs):
  mutated = np.copy(inputs)
  for i in xrange(len(inputs)):
    if mutation_probability > random.random():
      mutated[i] += random.uniform(-5, 5)
  return mutated

def selection(generation):
  #Individuals represented as (inputs, fitness)
  #Tournament Selection
  selection = random.sample(generation, tournament_size)
  return min(selection, key=lambda x: x[1])[0]

def crossover(parent1, parent2, method='one-point'):
  child1, child2 = np.copy(parent1), np.copy(parent2)
  if method == 'one-point':
    crossover_point = random.randint(1, len(child1)-1)
    tmp1 = child1[:crossover_point].copy()
    tmp2 = child2[:crossover_point].copy()
    child2[:crossover_point] = tmp1
    child1[:crossover_point] = tmp2
    return child1, child2
  elif method == 'two-point':
    first_point = random.randint(1, len(child1) - 2)
    second_point = random.randint(first_point, len(child1) - 1)
    tmp = child1[first_point:second_point].copy()
    child1[first_point:second_point] = child2[first_point:second_point].copy()
    child2[first_point:second_point] = tmp
    return child1, child2
  elif method == 'uniform':
    assert 0 <= mixing_ratio <= 1, str(mixing_ratio) + ' is not between 0 and 1'
    for i in xrange(len(child1)):
      if mixing_ratio > random.random():
        child1[i], child2[i] = child2[i], child1[i]
    return child1, child2
  elif method == 'arithmetic':
    alpha = random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2
  assert False, method + ' is not a crossover method.'

def create_initial_generation(range, func):
  #Evaluate individuals as adding them to generation
  generation = []
  for _ in xrange(generation_size):
    individual = np.random.uniform(range[0], range[1], 30)
    individual_fitness = fitness(individual, func)
    generation.append((individual, individual_fitness))
  return generation

def step_generation(generation, func, crossover_method):
  #Using selection and mutation
  #Also use elitism
  new_generation = []
  new_generation.extend([inputs for inputs in heapq.nsmallest(elite, generation, key=lambda x: x[1])])
  while len(new_generation) != len(generation):
    parent1 = selection(generation)
    parent2 = selection(generation)
    child1, child2 = crossover(parent1, parent2, crossover_method)
    child1, child2 = mutation(child1), mutation(child2)
    new_generation.append((child1, fitness(child1, func)))
    if len(new_generation) < len(generation):
      new_generation.append((child2, fitness(child2, func)))
  return new_generation

def make_graph(worst, average, best, func, crossover_method):
  plt.plot(range(0, len(worst)), worst, 'r-',\
           range(0, len(average)), average, 'b-',\
           range(0, len(best)), best, 'g-')
  plt.xlabel('Generations')
  plt.ylabel('Result of ' + func + ' Function')
  plt.suptitle('Genetic Algorithm on ' + func + ' Function')
  title = 'GenSize: {0} Mutation: {1:.0f}%, Crossover: {2}, Elite: {3:.0f}%, Tournament Size: {4}'.format(generation_size, mutation_probability * 100, crossover_method, elite_percent * 100, tournament_size)
  plt.title(title, fontsize=10)
  plt.show()

def factor_generator(num):
  factors = [x for x in range(2,int(5000/2)+1) if 5000%x==0]
  while len(factors) > 1:
    a = factors.pop()
    b = factors.pop(0)
    yield a, b
    yield b, a
  if len(factors) == 1:
    yield factors[0], factors[0]

for key, c_method in itertools.product(ranges, crossover_methods):
  results = np.empty(100)
  for t in xrange(100):
    #Create Generation
    #Get top, avg, worst to add to plot
    #Step generation
    #Repeat n times
    generation = create_initial_generation(ranges[key], key)
    best_fit = []
    avg_fit = []
    worst_fit = []
    for i in xrange(1, num_of_generations):
      worst_fit.append(max(generation, key=lambda x: x[1])[1])
      avg_fit.append(sum([individual[1] for individual in generation])/len(generation))
      best_fit.append(min(generation, key=lambda x: x[1])[1])
      generation = step_generation(generation, key, c_method)
    results[t] = min(generation, key=lambda x: x[1])[1]
  mean = sum(results) / 100.0
  error = np.std(results) / 100.0**.5
  print key, c_method, 'Mean:', mean, 'Error:', error