import numpy as np
from itertools import combinations
import math
import random
import matplotlib.pyplot as plt

ranges = {'Sphere': (-5.12, 5.12), 'Rosenbrock': (-2.048, 2.048), 'Rastrigin': (-5.12, 5.12), 'Schwefel': (-512.03, 511.97), 'Ackley': (-30, 30), 'Griewangk': (-600, 600)}
key = 'Schwefel'

#For neighbor generation
step_size = .25       #Amount to move variable in either direction
num_of_variables = 5  #Number of variables to change

#For simulated annealing
alpha = 0.99    #Usually a value between 0.8 and 0.99
repeats = 100   #Number of repeats on a certain temperature

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

def get_neighbor(inputs):
  neighbor = np.copy(inputs)
  variables = random.sample(range(len(inputs)), random.randint(1, num_of_variables+1))
  for variable in variables:
    neighbor[variable] += random.choice((-step_size, step_size))
  return neighbor

def hill_climb(range, func):
  #Creates pure random from range of values
  #Picks a random input
  #Checks up and down a certain step size for improvement
  #Selects the biggest improvement
  #Ends after no change for 200 iterations
  fitnesses = []
  inputs = np.random.uniform(range[0], range[1], 30)
  curr_fitness = fitness(inputs, func)
  streak = 0
  while streak < 200:
    '''point = random.randint(0, len(inputs) - 1)
    #Up movement
    tmp_up = np.copy(inputs)
    #print 'tmp_up[point]:', tmp_up[point]
    new_value = tmp_up[point] + step_size
    #print 'new_value: ', new_value
    if new_value <= range[1]:
      tmp_up[point] = new_value
    up_fitness = fitness(tmp_up, func)
    
    #Down movement
    tmp_down = np.copy(inputs)
    #print 'tmp_down[point]:', tmp_down[point]
    new_value = tmp_down[point] - step_size
    #print 'new_value: ', new_value
    if new_value <= range[1]:
      tmp_down[point] = new_value
    down_fitness = fitness(tmp_down, func)'''
    
    new_inputs = get_neighbor(inputs)
    new_fitness = fitness(new_inputs, func)
    
    if new_fitness < curr_fitness:
      curr_fitness = new_fitness
      inputs = new_inputs
      streak = 0
    else:
      streak += 1
    '''if up_fitness < curr_fitness:
      curr_fitness = up_fitness
      inputs = tmp_up
    if down_fitness < curr_fitness:
      curr_fitness = down_fitness
      inputs = tmp_down
    
    if curr_fitness == past_fitness:
      streak += 1
    else:
      streak = 0'''
    #past_fitness = curr_fitness
    fitnesses.append(curr_fitness)
  return inputs, curr_fitness, fitnesses

def temperature(a):
    T = 1.0
    T_min = 0.00001
    while T > T_min:
      yield T
      T *= a

def acceptance(e, e_prime, t):
  if e_prime < e:
    return 1
  else:
    return math.exp(-(e_prime - e)/t)

def simulated_annealing(range, func):
  #Creates pure random from range of values
  #Picks a random input
  #Selects n random variables
  #If improvement is found, take it
  #Otherwise, possibly take improvement based on acceptance def
  #Repeat n times before moving it down to a new temperature
  #Ends after minimum temperature is reached
  fitnesses = []
  inputs = np.random.uniform(range[0], range[1], 30)
  curr_fitness = fitness(inputs, func)
  t_generator = temperature(alpha)
  for t in t_generator:
    for _ in xrange(0, repeats):
      #Generate neighbor
      """point = random.randint(0, len(inputs)-1)
      direction = random.choice((step_size, -step_size))
      new_inputs = np.copy(inputs)
      new_inputs[point] += direction"""
      new_inputs = get_neighbor(inputs)
      new_fitness = fitness(new_inputs, func)
      if acceptance(curr_fitness, new_fitness, t) > random.random():
        inputs = new_inputs
        curr_fitness = new_fitness
      fitnesses.append(curr_fitness)
  return inputs, curr_fitness, fitnesses

def make_graph(fitnesses, method, func):
  plt.plot(fitnesses)
  plt.xlabel('Iterations')
  plt.ylabel('Result of ' + func + ' Function')
  plt.title(method + ' on ' + func + ' Function')
  plt.show()

best_fitness = 100000
best_inputs = None
for _ in xrange(0, 100):
  inputs, fit, plot = hill_climb(ranges[key], key)
  if fit < best_fitness:
    best_fitness = fit
    best_inputs = inputs
  print fit
print best_inputs, fitness
#make_graph(plot, 'Hill Climbing', key)
#inputs, fit, plot = simulated_annealing(ranges[key], key)
#print 'Best Inputs:', (inputs, fit)
#make_graph(plot, 'Simulated Annealing', key)