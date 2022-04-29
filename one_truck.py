import numpy
import random
import matplotlib.pyplot as plt

''' Parameters '''
# Population size
pop_size = 10000
# Number of counties
county_number = 19
# Number of iterations to run the algorithm
it = 11
# Distance between two tested weights
precision_of_pareto = 50
# Number of breeds at each iteration
children_fraction_in_population = 0.75
amount_of_children = round(children_fraction_in_population * pop_size)


# We want 6 children, we need 4 parents, because :
# Parent 1 breed 3 times (with 2, 3 and 4),
# Parent 2 breed 2 more times (with 3 and 4),
# Parent 3 breed 1 more time (only with 4),
# Parent 4 breed 0 more times
def how_many_parents(number_of_children):
    number_of_children_left = number_of_children
    iteration_number = 1
    while number_of_children_left > 0:
        number_of_children_left -= iteration_number
        iteration_number += 1

    return iteration_number  # number of parents needed to breed number_of_children times


print(amount_of_children)
print(how_many_parents(amount_of_children))

''' Variables '''
money_matrix = [179797, 131547, 118920, 96501, 86675, 82742,
                56496, 55925, 52417, 49715, 48008, 41789, 41588,
                33970, 27097, 25195, 25172, 24817, 21961]


# Import the distance matrix from a .csv file
def import_matrix(csv_name):
    matrix = numpy.genfromtxt(csv_name, delimiter=';')
    return matrix


distance_matrix = import_matrix("resources/matrix.csv")


# Generate a random initial population of counties with pop_size individuals
def genetic_generate_init():
    pop = []
    for i in range(pop_size):
        pop.append(list(numpy.random.permutation(county_number)))
    return pop


# Cross over two parents to give one child
def crossover(parent1, parent2):
    # c = crossover spots
    c1 = random.randint(0, county_number - 1)
    c2 = random.randint(c1, county_number - 1)

    child = []
    # First half of the child comes from the first parent
    for i in range(0, c1):
        child.append(parent1[i])

    # Second half of the child comes from the second parent, in order of appearance
    for i in range(c1, c2):
        for j in range(0, county_number):
            if parent2[j] not in child:
                child.append(parent2[j])
                break

    for i in range(c2, county_number):
        for j in range(0, county_number):
            if parent2[j] not in child:
                child.append(parent2[j])
                break

    return child


# Mutation : switch two random items of the list
def mutate(individual):
    r1 = random.randint(0, county_number - 1)
    r2 = random.randint(0, county_number - 1)
    temp = individual[r1]
    individual[r1] = individual[r2]
    individual[r2] = temp

    return individual


# Returns the total distance travelled for the individual
def find_total_dist(individual):
    dist = 0
    for i in range(county_number - 1):
        dist += distance_matrix[max(individual[i], individual[i + 1])][min(individual[i], individual[i + 1])]
        # print("Distance : " + str(dist))
    return dist


# Returns the distance travelled for the individual weighted by the amount of money carried
def find_weighted_dist(individual):
    w_dist = 0
    carry = 0
    for i in range(county_number - 1):
        carry += money_matrix[individual[i]]
        w_dist += distance_matrix[max(individual[i], individual[i + 1])][min(individual[i], individual[i + 1])] * carry
        # print("Weigthed Distance : " + str(w_dist))
    return w_dist


def score(individual, weight):
    return weight * find_weighted_dist(individual) + (1 - weight) * find_total_dist(individual) * 440000


def sorted_population_score(pop, weight):
    """
    :param pop: population to determine the score of
    :param weight: importance of criteria 1 against criteria 2 (in [0,1])
    :return: a list of tuples (individual, score) sorted by score in descending order
    """
    list_score = []

    for ind in pop:
        list_score.append((ind, score(ind, weight)))

    return sorted(list_score, key=lambda x: x[1], reverse=False)


def breed(list_score):
    n_parents = how_many_parents(amount_of_children)
    new_population = []

    # Add the parents to the new population
    for parent_index in range(n_parents):
        # new_population.append(list_score[parent_index][0])
        for ind_index in range(parent_index, n_parents):
            if len(new_population) < pop_size:
                new_population.append(crossover(list_score[parent_index][0],
                                                list_score[ind_index][0]))

    if len(new_population) < pop_size:
        for relative_ind_index in range(pop_size - len(new_population)):
            new_population.append(mutate(list_score[n_parents + relative_ind_index][0]))

    for index in range(pop_size):
        if random.randint(0, 10) == 5:
            new_population[index] = mutate(new_population[index])
    return new_population


def filter_value(x, y):
    x_pareto = []
    y_pareto = []

    for i in range(len(x)):
        opt = True
        for j in range(len(x)):
            if x[j] > x[i] and y[j] > y[i]:
                opt = False
        if opt:
            x_pareto.append(x[i])
            y_pareto.append(y[i])

    return x_pareto, y_pareto


if __name__ == '__main__':
    distance_matrix = import_matrix("resources/matrix.csv")
    # print(distance_matrix)
    results_ind = []
    results_scores = []

    for alpha in range(0, precision_of_pareto + 1):
        for x in range(10):
            weight = alpha / precision_of_pareto
            population = genetic_generate_init()
            print("Iteration (alpha) : " + str(alpha) + ", sub-iteration : " + str(x) + ", weight = " + str(weight))
            for i in range(it):
                population = breed(sorted_population_score(population, weight))
                res = sorted_population_score(population, weight)
                print("New Score : " + str(res[0][1]))

            results_ind.append(res[0][0])
            # print(str(res[0][0]) + " " + str(res[0][1]))
            results_scores.append([find_total_dist(results_ind[-1]), find_weighted_dist(results_ind[-1])])

    x_val = [x[0] for x in results_scores]
    y_val = [x[1] for x in results_scores]
    x_p, y_p = filter_value(x_val, y_val)
    plt.plot(x_p, y_p, 'or')
    plt.show()

'''
    population = genetic_generate_init()
    print(population)
    sorted = sorted_population_score(population, 0.5)
    for ind in sorted:
        print(str(ind[0]) + " score is : " + str(ind[1]))
    print(breed(sorted))
    # print(breed(sorted_population_score(population, 0.5)))
'''

