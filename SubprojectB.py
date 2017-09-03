import numpy as np
from itertools import combinations
import math
import random
import matplotlib.pyplot as plt
import heapq

ranges = {'Sphere': (-5.12, 5.12), 'Rosenbrock': (-2.048, 2.048), 'Rastrigin': (-5.12, 5.12), 'Schwefel': (-512.03, 511.97), 'Ackley': (-30, 30), 'Griewangk': (-600, 600)}
key = 'Sphere'

#For Genetic Algorithm
generation_size = 100
num_of_generations = 1000
mutation_probability = 1.0 / 30.0
elite = int(.05 * generation_size)
tournament_size = 5

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

def create_initial_generation(range, func):
  #Evaluate individuals as adding them to generation
  generation = []
  for _ in xrange(generation_size):
    individual = np.random.uniform(range[0], range[1], 30)
    individual_fitness = fitness(individual, func)
    generation.append((individual, individual_fitness))
  return generation

def step_generation(generation, func):
  #Using selection and mutation
  #Also use elitism
  new_generation = []
  new_generation.extend([inputs for inputs in heapq.nsmallest(elite, generation, key=lambda x: x[1])])
  for _ in xrange(elite, len(generation)):
    inputs = selection(generation)
    inputs = mutation(inputs)
    new_generation.append((inputs, fitness(inputs, func)))
  return new_generation

def make_graph(worst, average, best, func):
  plt.plot(range(0, len(worst)), worst, 'r-',\
           range(0, len(average)), average, 'b-',\
           range(0, len(best)), best, 'g-')
  plt.xlabel('Generations')
  plt.ylabel('Result of ' + func + ' Function')
  plt.title('Partial Genetic Algorithm on ' + func + ' Function')
  plt.show()

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
  print i
  generation = step_generation(generation, key)

print 'Best:', min(generation, key=lambda x: x[1])
make_graph(worst_fit, avg_fit, best_fit, key)