"""
						SOLUTION TO BANK-LENDING PROBLEM WITH GENETIC ALGORITHM IMPLEMENTATION
						Author : Mintu Agarwal
"""

import random
import matplotlib.pyplot as plt
from copy import deepcopy 

D = 60 #given constraints on credit available
K = 0.15 #reserved ratio of the deposit


# individual_list data provided
loan_size = [10, 25, 4, 11, 18, 3, 17, 15, 9, 10]
interest = [0.021, 0.022, 0.021, 0.027, 0.025, 0.026, 0.023, 0.021, 0.028, 0.022]
rating = ["AAA", "BB", "A", "AA", "BBB", "AAA", "BB", "AAA", "A", "A"]
loss = [0.0002, 0.0058, 0.0001, 0.0003, 0.0024, 0.0002, 0.0058, 0.0002, 0.001, 0.001]

# GA is a population-based algorithm
population_size = 60
string_length = len(loan_size)

""" 			SUPPORTING FUNCTIONS and INITIAL SOLUTION GENERATION 			"""

# stores the current population of individuals
individual_list = []

# generate individual/binary-string with random bits at each position
def generate_chromosome():
    new_string = []

    for i in range(string_length):
        new_string.append(random.randint(0, 1))
    return new_string

#checks if a individual(chromosome) satifies the total credit constraint
def is_valid_string(individual): 
	loan_sum = 0

	for i in range(string_length):
		loan_sum += individual[i]*loan_size[i]

	return loan_sum <= (1-K)*D


# generating intial generation of individuals
for i in range(population_size):
	new_indi = generate_chromosome() 		# binary string/array representing a individual
	
	while not is_valid_string(new_indi): 	# update while not a valid random chromosome is obtained
		new_indi = generate_chromosome()

	individual_list.append(new_indi) 		# generated individual appended to the population

Rt = 0.01
Rd = 0.009


# find fitness of a individual(solution)
def fitness(individual):

	V, omega, beta, loan_sum, sum_loss = 0, 0, Rd*D, 0, 0

	for i in range(string_length):

		V += individual[i]*(interest[i]*loan_size[i]-loss[i])

		omega += individual[i]*(Rt*((1-K)*D-loan_size[i])) #total transaction cost of the expected lending decision

		loan_sum += individual[i]*loan_size[i]

		sum_loss += individual[i]*loss[i]


	fitness_value = V + omega - beta - sum_loss # Fx calculated

	return (fitness_value)


# GA optimization parameters

no_of_generations = 60 	# equivalent to number of iterations
P_c = 0.8				# crossover probability
P_m = 0.006				# mutation probability
P_reproduction = 0.194 	#reproduction ratio


""" DEFINING GENETIC FUNCTIONS """

# selection of parent pool
def roulette_wheel_selection(population):
	
	fitness_population = []			# stores fitness values of all individuals in the population

	for i in range(len(population)):
		fitness_population.append(fitness(population[i]))

	sum_fitness = sum(fitness_population)
	fitness_population = [i/sum_fitness for i in fitness_population] # fitness normalization
	
	for i in range(1,len(fitness_population)):
		fitness_population[i] = fitness_population[i] + fitness_population[i-1]
	
	rand1, rand2 = random.random(), random.random()
	p1, p2 = None, None
	
	for i in range(len(fitness_population)):
		if rand1 < fitness_population[i] and p1 is None:
			p1 = population[i]
		if rand2 < fitness_population[i] and p2 is None:
			p2 = population[i]

	return p1, p2

def elite(population):
	fitness_population=[]

	no_of_elite = int(len(population)*P_reproduction) + 1		# number of elite individuals to be stored
	
	for i in range(len(population)):
		fitness_indi = fitness(population[i])
		fitness_population.append([fitness_indi, population[i]])

	fitness_population = sorted(fitness_population, reverse=True, key = lambda x : x[0])	# individuals sorted based on fitness

	elite_individuals = []
	for i in range(no_of_elite):
		elite_individuals.append(fitness_population[i][1])
	return elite_individuals

def single_point_crossover(p1, p2):
	# to avoid generating children same as parent, random index must not lie at either end of string

	pivot = random.randint(1,len(p1)-2) # random index generated for single-point crossover

	# crossover operation
	c1 = p1[0:pivot] + p2[pivot:]
	c2 = p2[0:pivot] + p1[pivot:]

	return c1, c2


def mutation(parent):
	temp_parent = deepcopy(parent)

	for i in range(len(temp_parent)):
		if random.random() < P_m:
			temp_parent[i]= 1 - temp_parent[i]	# flipping the bit randomly

	#mutated individual returned
	return temp_parent

""" MAIN """

individual_list_old = individual_list
fittest_individual =[]

for i in range(no_of_generations):

	individual_list_new = elite(individual_list_old)
	fittest_individual.append(fitness(individual_list_new[0]))
	
	while(len(individual_list_new)!= len(individual_list_old)):	
		parent1,parent2 = roulette_wheel_selection(individual_list_old)
		child1 = None
		child2 = None
		
		if random.random() < P_c:
			child1, child2 = single_point_crossover(parent1,parent2)
		else:
			child1, child2 = parent1,parent2

		child1 = mutation(child1)
		child2 = mutation(child2)

		#validating the newly generated individual
		while not is_valid_string(child1):
			child1 = generate_chromosome()

		while not is_valid_string(child2):
			child2 = generate_chromosome()

		individual_list_new.append(child1)
		individual_list_new.append(child2)

	individual_list_old = individual_list_new

"""			RESULTS	AND PLOTS		"""

individual_list_current = elite(individual_list_old)
print("Best Individual found is ")
print(individual_list_current[0])
print("With fitness Value = ")
print(fittest_individual[no_of_generations - 1])
plt.plot(range(no_of_generations),fittest_individual)
plt.show()