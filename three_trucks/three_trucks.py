"""
INFO-H3000 - Recherche opérationnelle
Liam Fallik, Quentin Roels

This script is the main script of the project. It calculates the optimal route for m trucks going in n cities with
m and n as parameters.
The output of this script is a plot and an ordered list of cities (see report for more information)
"""

from random import randint
import matplotlib.pyplot as plt
import numpy
import csv

""" Parameters """
# Population size
pop_size = 10000
# Number of counties
number_of_counties = 19
# Number of salesmen
number_of_truck = 3
# Number of iterations to run the algorithm
it = 10
# Trying multiple times on the same weight
tries_on_same_weight = 1
# Distance between two tested weights
precision_of_pareto = 100
# Number of breeds at each iteration
children_fraction_in_population = 0.9
amount_of_children = round(children_fraction_in_population * pop_size)
# Mutation chance (one over mut)
mut = 20
# Large number
INFINITY = 100000000
# Distance matrix (will be imported when calling import_matrix)
distance_matrix = []


""" Constants """
# Population of each city in the same order as distance_matrix
# Based on data from the domestic federal public service of Belgium
# https://www.ibz.rrn.fgov.be/fileadmin/user_upload/fr/pop/statistiques/population-bevolking-20190101.pdf
population_cities = [179797, 131547, 118920, 96501, 86675, 82742,
                     56496, 55925, 52417, 49715, 48008, 41789, 41588,
                     33970, 27097, 25195, 25172, 24817, 21961]
# Each person pays 0.70 € per month in cash that needs to be collected
money_cities = [element * 0.7 for element in population_cities]
# Total amount in the system for checking the money constraint
money_total = sum(money_cities)
# Distance list between National Bank and Counties
dist_to_bank = [1.0, 3.0, 3.0, 2.5, 3.6, 7.0, 5.9, 6.0, 5.8, 3.7, 4.1, 5.7, 5.3, 7.0, 1.6, 7.3, 9.1, 7.0, 4.7]


""" Functions """


# If we want 6 children, we need 4 parents, because :
# Parent 1 breeds 3 times (with 2, 3 and 4),
# Parent 2 breeds 2 more times (with 3 and 4),
# Parent 3 breeds 1 more time (only with 4),
# Parent 4 breeds 0 more times
def how_many_parents(number_of_children):
    """
    Computes the number of parents needed to breed number_of_children times
    :param number_of_children: a parameter of the program
    :return: the number of parents
    """
    number_of_children_left = number_of_children
    number_of_parents = 2
    while number_of_children_left > 0:
        number_of_children_left -= number_of_parents - 1
        number_of_parents += 1

    return number_of_parents


def import_matrix(csv_name):
    """
    Import the distance matrix from a .csv file
    :param csv_name: the path to the .csv file
    :return: a matrix that gives the distances between every county
    """
    matrix = numpy.genfromtxt(csv_name, delimiter=';')
    return matrix


def genetic_generate_init():
    """
    Generate a random initial population of counties with pop_size individuals according to Carter and Ragsdale's
    method.
    Individual format: [1, 2, 3, 4, ... , a, b, c, ...]
                        cities            salesmen
    With a, b, c, ... = the number of cities visited by each one of the salesmen with a + b + c + ... = the total number
    of counties
    :return: the whole population as a matrix (list of lists)
    """
    pop = []
    while len(pop) < pop_size:
        # Initialize an individual with negative values (to be filled)
        ind = [-1] * (number_of_counties + number_of_truck)
        cities_left = number_of_counties

        # Chose the number of cities visited by each truck of the individual and place them at the end of the
        # chromosome.
        for truck_number in range(number_of_truck - 1):
            ind[truck_number - number_of_truck] = randint(1, cities_left - (number_of_truck - truck_number))
            cities_left -= ind[truck_number - number_of_truck]
        ind[-1] = cities_left

        # Place the biggest (main) cities according to the constraint: each truck can only visit one main city
        biggest_counties = list(numpy.random.permutation(number_of_truck))
        offset = 0
        for truck_number in range(number_of_truck):
            place = randint(0, ind[truck_number - number_of_truck] - 1)
            ind[place + offset] = biggest_counties[truck_number]
            offset += ind[truck_number - number_of_truck]

        # Place the other cities randomly
        others = list(numpy.random.permutation(number_of_counties - number_of_truck))
        for j in range(number_of_counties):
            if ind[j] < 0:
                ind[j] = others.pop() + number_of_truck

        # Check the constraints and append the individual if it satisfies both
        if is_biggest_county_constraint_verified(ind) and is_half_amount_constraints_verified(ind):
            pop.append(ind)

    return pop


def make_list_from_csv(path: str):
    """
    Parse a CSV to a matrix
    :param path: the path of the CSV file
    :return: the data as a matrix
    """
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def calculate_weight_constant() -> float:
    """
    Calculate the weight by dividing the sum of the weighted distances by the sum of the distances of the best
    performing solutions. This weight allows us to compare the distance and the weighted distance of an individual, as
    the latter is much bigger than the first
    @return: The weight
    """
    best_pop = make_list_from_csv("../results/To Keep/New/all_data_filtered")
    sum_weight = 0
    sum_distance = 0
    for ind in best_pop:
        sum_weight += find_weighted_dist(ind)
        sum_distance += find_total_dist(ind)

    print("Weight factor : " + str(sum_weight/sum_distance))
    return sum_weight / sum_distance


def is_biggest_county_constraint_verified(ind):
    """
    This function tests if an individual satisfies the constraint that the main cities are each visited by a
    different truck
    :param ind: The individual to be tested
    :return: True if the constraint is satisfied, False if not
    """
    constraints_verified = True
    offset = 0
    for truck_number in range(number_of_truck):
        truck_passed_biggest = False
        for county in range(ind[truck_number - number_of_truck]):
            if ind[offset + county] in range(number_of_truck) and truck_passed_biggest:
                constraints_verified = False
            elif ind[offset + county] in range(number_of_truck) and not truck_passed_biggest:
                truck_passed_biggest = True
        offset += ind[truck_number - number_of_truck]

    return constraints_verified


def is_half_amount_constraints_verified(ind):
    """
    This function verifies if an individual satisfies the constraint that it can't transport more than half of the total
    amount at once
    :param ind: The individual to be tested
    :return: True if the constraint is satisfied, False if not
    """
    constraint_verified = True
    offset = 0
    for truck_number in range(number_of_truck):
        total_money = 0
        for i in ind[offset:offset + ind[truck_number - number_of_truck] - 1]:
            total_money += money_cities[i]
        if total_money > 0.5 * money_total:
            constraint_verified = False
        offset += ind[truck_number - number_of_truck]
    return constraint_verified


def crossover(mom, dad):
    """
    Cross two parent chromosomes to get one child
    Crossover method by S. Yuan et al.: https://doi.org/10.1016/j.ejor.2013.01.043
    :param mom: the first parent
    :param dad: the second parent
    :return: the child
    """
    total_unsaved_genes = 19
    total_saved_genes = 0

    saved_genes = []
    unsaved_genes = []

    child_parts = []

    # First step: choose the genes to keep from the mother
    for truck_number in range(number_of_truck):
        new_child = []
        ref_to_truck = truck_number - number_of_truck  # index in second part of the gene
        assigned_county = mom[truck_number - number_of_truck]  # value in second part of gene
        segment_size = randint(0, assigned_county - 1)  # size of the segment we retrieve

        starting_position = 0
        if assigned_county > segment_size:
            starting_position = randint(0, assigned_county - segment_size - 1)

        for index in range(segment_size):
            offset = 0  # offset to the index of the first county visited by the truck
            for truck in range(truck_number):
                offset += mom[truck - number_of_truck]
            # Copy each gene from the gene segment for a Child
            new_child.append(mom[offset + starting_position + index])
            # Copy each gene into savedGenesPool
            saved_genes.append(mom[offset + starting_position + index])

        total_saved_genes = total_saved_genes + segment_size
        child_parts.append(new_child)

    total_unsaved_genes = number_of_counties - total_saved_genes

    # Second step: fill the individual with genes from the dad
    number_to_add_per_truck = []
    offset = 0
    for truck_number in range(number_of_truck):
        number_to_add_in_truck = dad[truck_number - number_of_truck]
        cities_to_add_to_truck = 0
        for i in range(number_to_add_in_truck):
            if dad[offset + i] not in saved_genes:
                unsaved_genes.append(dad[offset + i])
                cities_to_add_to_truck += 1

        number_to_add_per_truck.append(cities_to_add_to_truck)
        offset += number_to_add_in_truck

    index = 0
    for truck_number in range(number_of_truck):
        number_to_save = number_to_add_per_truck[truck_number]

        # According to the order of the unsaved genes in the first
        # part of Dad’s chromosome, add the randomly generated
        # number of genes to the Child;
        for i in range(number_to_save):
            child_parts[truck_number].append(unsaved_genes[index])
            total_unsaved_genes -= 1
            index += 1

    # Third step: create the child
    child = [0] * (number_of_counties + number_of_truck)
    index = 0
    for truck_number in range(len(child_parts)):
        assigned_county = len(child_parts[truck_number])
        # Second part of the chromosome
        child[truck_number - number_of_truck] = assigned_county
        # First part of the chromosome
        for county in range(assigned_county):
            child[index] = child_parts[truck_number][county]
            index += 1

    return child


def mutate(ind):
    """
    Mutates an individual with a random chance by switching two cities and two numbers from the trucks
    :param ind: The individual to mutate
    """
    # Backup the individual in case its mutation doesn't satisfy the constraints
    ind_backup = ind.copy()
    if randint(0, mut) == 0:
        # switch two numbers from the "cities" section
        pos1 = randint(0, number_of_counties - 1)
        pos2 = randint(0, number_of_counties - 1)
        ind[pos1], ind[pos2] = ind[pos2], ind[pos1]
        # Switch two numbers from the "trucks" section
        pos3 = randint(0, number_of_truck - 1)
        pos4 = randint(0, number_of_truck - 1)
        ind[number_of_counties + pos3], ind[number_of_counties + pos4] = \
            ind[number_of_counties + pos4], ind[number_of_counties + pos3]
    if not is_half_amount_constraints_verified(ind) or not is_biggest_county_constraint_verified(ind):
        ind = ind_backup.copy()

    return ind


def find_total_dist(ind):
    """
    Calculate the total distance travelled by the trucks for an individual. This distance comprises the distances
    between each city visited by a truck but also the distance travelled to and from the central bank (at the start
    and end of the trip)
    :param ind: the individual
    :return: the distance
    """
    dist = 0
    offset = 0
    for truck_number in range(number_of_truck):
        assigned_county = ind[truck_number - number_of_truck]

        dist += dist_to_bank[ind[offset]]  # Start from the Bank
        dist += dist_to_bank[ind[offset + assigned_county - 1]]  # End at the Bank
        for i in range(assigned_county - 1):
            dist += distance_matrix[max(ind[offset + i], ind[offset + i + 1])][min(ind[offset + i], ind[offset + i + 1])]

        offset += assigned_county

    return dist


# Returns the distance travelled for the individual weighted by the amount of money carried
def find_weighted_dist(ind):
    """
    Calculate the total weighted distance travelled by the trucks for an individual. This distance comprises the
    distances between each city visited by a truck multiplied by the amount of money carried on that part of the trip,
    as well as the distance between the last stop and the central bank multiplied by the total amount transported.
    :param ind: The individual
    :return: The weighted distance
    """
    w_dist = 0
    carry = 0
    offset = 0
    for truck_number in range(number_of_truck):
        assigned_county = ind[truck_number - number_of_truck]

        for i in range(assigned_county - 1):
            carry += money_cities[ind[offset + i]]
            w_dist += distance_matrix[max(ind[offset + i], ind[offset + i + 1])][min(ind[offset + i], ind[offset + i + 1])] * carry
        w_dist += dist_to_bank[ind[offset + assigned_county - 1]] * carry  # End at the Bank

        offset += assigned_county

    return w_dist


def score(ind, weight, weight_balance_cst):
    """
    Calculates the fitness of an individual
    :param ind: The individual
    :param weight: A number between 0 and 1 describing how much of each criterion we want to prioritize
    :param weight_balance_cst: A constant to balance the impact of both criteria in the calculation
    :return: The score
    """
    return weight * find_weighted_dist(ind) + (1 - weight) * find_total_dist(ind) * weight_balance_cst


def sorted_population_score(pop: list, weight: float, weight_balance: float):
    """
    Sorts the population by descending score
    :param pop: The population going to be sorted
    :param weight: importance of criteria 1 against criteria 2 (in [0,1], 0 means only the first criteria is
                   applied, 1 means only the second is applied)
    :param weight_balance: A constant to balance the impact of both criteria in the calculation letting us compare
                           them
    :return: a list of tuples (individual, score) sorted by score in descending order
    """
    list_score = []

    for ind in pop:
        list_score.append((ind, score(ind, weight, weight_balance)))

    return sorted(list_score, key=lambda k: k[-1], reverse=False)


def breed(list_score, n_parents):
    """
    Breed the entire population with itself according to the parameters
    :param list_score: The population sorted by descending score
    :param n_parents: The number of parents we need to breed
    :return: The new generation of individuals
    """
    new_population = []

    # Add the parents to the new population
    for parent_index in range(n_parents):
        new_population.append(list_score[parent_index][0])
        # Crossover the parents
        for ind_index in range(parent_index, n_parents):
            for i in range(2):  # Add Sister and Brother
                if len(new_population) < amount_of_children:
                    child = crossover(list_score[parent_index][0], list_score[ind_index][0])
                    while not is_half_amount_constraints_verified(child) or \
                            not is_biggest_county_constraint_verified(child):
                        child = crossover(list_score[parent_index][0], list_score[ind_index][0])

                    new_population.append(child)

    # If the new population doesn't have enough individuals because some children didn't satisfy the constraints, add
    # some new ones
    if len(new_population) < pop_size:
        for relative_ind_index in range(pop_size - len(new_population)):
            new_population.append(mutate(list_score[n_parents + relative_ind_index][0]))

    for index in range(pop_size):
        new_population[index] = mutate(new_population[index])
    return new_population


def filter_pareto(x, y):
    """
    Filter the outputs to get a pareto optimal curve (meaning each point shouldn't be beaten on both criteria by
    another)
    :param x: The distances
    :param y: The weighted distances
    :return: Filtered lists for the abscissa and ordinate of the graph
    """
    x_pareto = []
    y_pareto = []

    for i in range(len(x)):
        opt = True
        for j in range(len(x)):
            if x[j] < x[i] and y[j] < y[i]:
                opt = False
        if opt:
            x_pareto.append(x[i])
            y_pareto.append(y[i])

    return x_pareto, y_pareto


def simulate_one_weight(weight: float):
    results_ind = []
    results_scores = []

    for tries in range(tries_on_same_weight):
        population = genetic_generate_init()
        weight_constant = calculate_weight_constant()
        number_of_parents = how_many_parents(amount_of_children)
        print("Iteration : " + str(tries) + ", Weight = " + str(weight))
        res = [[[INFINITY]]]
        for i in range(it):
            population = breed(sorted_population_score(population, weight, weight_constant), number_of_parents)
            res = sorted_population_score(population, weight, weight_constant)
            # print("New Score : " + str(res[0][1]))

        results_ind.append(res[0][0])
        print("score : " + str(score(res[0][0], weight, weight_constant)))
        results_scores.append([find_total_dist(results_ind[-1]), find_weighted_dist(results_ind[-1])])

    x_val = [lst[0] for lst in results_scores]
    y_val = [lst[1] for lst in results_scores]
    return x_val, y_val, results_ind


def simulate_mtsp():
    """
    The main method of this script. Iterates a breed and mutation on the population a number of times then returns the
    best individuals
    @return: the distance, weighted score of each individual and the individual themselves (one individual per weight)
    """
    results_ind = []
    results_scores = []
    weight_constant = calculate_weight_constant()

    # Try every weight between 0 and 1 with a step of precision_of_pareto
    for alpha in range(0, precision_of_pareto + 1):
        # Try several times on the same weight to be sure to get the best possible answer
        for x in range(tries_on_same_weight):
            weight = alpha / precision_of_pareto
            population = genetic_generate_init()
            number_of_parents = how_many_parents(amount_of_children)
            print("Iteration (alpha) : " + str(alpha) + ", sub-iteration : " + str(x) + ", weight = " + str(weight))
            res = [[[INFINITY]]]
            # Breed and mutate the population [it] times
            for i in range(it):
                population = breed(sorted_population_score(population, weight, weight_constant), number_of_parents)
                res = sorted_population_score(population, weight, weight_constant)

            # Append the best individual to the results
            results_ind.append(res[0][0])
            print("score :      " + str(score(res[0][0], weight, weight_constant)))
            print("distance :   " + str(find_total_dist(res[0][0])))
            print("risk :       " + str(find_weighted_dist(res[0][0])))
            print("individual : " + str(res[0][0]))
            results_scores.append([find_total_dist(results_ind[-1]), find_weighted_dist(results_ind[-1])])

    # Get the list of distances and weighted distances of the best individuals on every weight
    x_val = [lst[0] for lst in results_scores]
    y_val = [lst[1] for lst in results_scores]
    return x_val, y_val, results_ind


def plot(x_in, y_in):
    """
    Plot the pareto optimal curve
    """
    x_p, y_p = filter_pareto(x_in, y_in)
    plt.plot(x_p, y_p, 'or')
    plt.show()


def save_csv(solutions, file_name: str):
    """
    Saves the solution as a csv file with each possible solution on the pareto optimal curve on a separate row
    :param solutions: the list of solutions returned by simulate_mtsp
    :param file_name: the name of the csv file
    """
    with open(file_name, 'w') as csv_file:
        for solution in solutions:
            writer = csv.writer(csv_file)
            writer.writerow(solution)


# Main method, run this with the desired parameters at the beginning of this script and the correct path for the input
# matrix.
if __name__ == '__main__':
    distance_matrix = import_matrix("../resources/matrix.csv")
    x, y, results = simulate_mtsp()

    # Save the csv file
    file_name = f"../results/{pop_size}pop_{number_of_counties}_{number_of_truck}_{it}it_{tries_on_same_weight}try_{precision_of_pareto}prec_{round(children_fraction_in_population * 100)}child.csv"
    save_csv(results, file_name)
