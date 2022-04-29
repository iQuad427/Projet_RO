from random import randint
import numpy

"""Parameters"""
# Population size
pop_size = 10000
# Number of counties
number_of_county = 19
# Number of salesmen
number_of_truck = 3
# Number of iterations to run the algorithm
it = 11
# Distance between two tested weights
precision_of_pareto = 50
# Number of breeds at each iteration
children_fraction_in_population = 0.75
amount_of_children = round(children_fraction_in_population * pop_size)

""" Variables """
# Based on data from the domestic federal public service of Belgium
# https://www.ibz.rrn.fgov.be/fileadmin/user_upload/fr/pop/statistiques/population-bevolking-20190101.pdf
population_cities = [179797, 131547, 118920, 96501, 86675, 82742,
                     56496, 55925, 52417, 49715, 48008, 41789, 41588,
                     33970, 27097, 25195, 25172, 24817, 21961]
# Each person pays 0.70 € per month in cash that needs to be collected
money_cities = [element * 0.7 for element in population_cities]

"""Functions"""


def import_matrix(csv_name):
    """
    Import the distance matrix from a .csv file
    :param csv_name: the path to the .csv file
    :return: a matrix that gives the distances between every county
    """
    matrix = numpy.genfromtxt(csv_name, delimiter=';')
    return matrix


distance_matrix = import_matrix("resources/matrix.csv")


def genetic_generate_init():
    """
    Generate a random initial population of counties with pop_size individuals according to Carter and Ragsdale's
    method.
    Individual format: [1, 2, 3, 4, ... , a, b, c, ...]
                        cities            salesmen
    With a, b, c, ... = the number of cities visited by each one of the salesmen
    :return: the whole population as a matrix (list of lists)
    """
    pop = []
    for i in range(pop_size):
        pop.append([0] * (number_of_county + number_of_truck))
        cities_left = number_of_county
        for j in range(1, number_of_truck):
            pop[i][-j] = randint(1, cities_left - number_of_truck + j)
            cities_left -= pop[i][-j]
        pop[i][-number_of_truck] = cities_left

        # place the 3 biggest cities according to the constraint
        three_biggest = list(numpy.random.permutation(number_of_truck))
        offset = 0
        for truck_number in range(number_of_truck):
            place = randint(0, pop[i][-(truck_number + 1)])
            pop[i][place + offset] = three_biggest[truck_number]
            offset += pop[i][-(truck_number + 1)]
    return pop


# Format of a solution : list = [1, 2, 3, 4, ... , a, b, c, ...]
#                                cities            salesmen
# with a, b, c the number of cities visited by each salesman


def crossover(mom, dad):
    """
    Cross two parent chromosomes to get two children
    Crossover method by S. Yuan et al.: https://doi.org/10.1016/j.ejor.2013.01.043
    :param mom: the first parent
    :param dad:
    :return: two children
    """
    print(f"mom : {mom}")
    print(f"dad : {dad}")
    total_unsaved_genes = 19
    total_saved_genes = 0

    saved_genes = []
    unsaved_genes = []

    child_parts = []

    print("\n FIRST STEP : KEEP FROM MOM \n")

    for truck_number in range(number_of_truck):
        new_child = []
        ref_to_truck = truck_number - number_of_truck  # index in second part of the gene
        assigned_county = mom[truck_number - number_of_truck]  # value in second part of gene
        segment_size = randint(0, assigned_county - 1)  # size of the segment we retrieve

        print(f"number of county assigned to truck number {truck_number} : {assigned_county}")
        print(f"segment size : {segment_size}")

        starting_position = 0
        if assigned_county > segment_size:
            starting_position = randint(0, assigned_county - segment_size - 1)

        print(f"starting position : {starting_position}")

        for index in range(segment_size):
            offset = 0  # offset to the index of the first county visited by the truck
            for truck in range(truck_number):
                offset += mom[truck - number_of_truck]
            # Copy each gene from the gene segment for a Child
            new_child.append(mom[offset + starting_position + index])
            # Copy each gene into savedGenesPool
            saved_genes.append(mom[offset + starting_position + index])

        total_saved_genes = total_saved_genes + segment_size
        print(f"new_child : {new_child}")
        child_parts.append(new_child)
        print(f"child_parts : {child_parts}")

    total_unsaved_genes = number_of_county - total_saved_genes

    print("\n SECOND STEP : COMPLETE FROM DAD \n")

    for i in range(number_of_county):
        if dad[i] not in saved_genes:
            unsaved_genes.append(dad[i])

    print(f"unsaved genes : {unsaved_genes}")

    index = 0
    for truck_number in range(number_of_truck):
        number_to_save = total_unsaved_genes
        if truck_number != number_of_truck - 1:
            # Randomly generate an integer number between 1
            # and totalUnsavedGenes for salesman m to add genes
            if total_unsaved_genes != 0:
                number_to_save = randint(1, total_unsaved_genes)

        print(f"truck number : {truck_number}")
        print(f"number to save : {number_to_save}")

        # According to the order of the unsaved genes in the first
        # part of Dad’s chromosome, add the randomly generated
        # number of genes to the Child;
        for i in range(number_to_save):
            child_parts[truck_number].append(unsaved_genes[index])
            total_unsaved_genes -= 1
            index += 1

        print(f"completed list : {child_parts}")

    print("\n THIRD STEP : CREATE CHILD \n")
    print(child_parts)

    child = [0] * (number_of_county + number_of_truck)
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

    # Based on the construction of the first part of the Sister’s
    # chromosome, calculate the number of assigned cities in the
    # second part of the chromosome.


if __name__ == '__main__':
    mom1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 10, 5, 4]
    dad1 = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 2, 7, 10]
    child1 = crossover(mom1, dad1)
    print(child1)
