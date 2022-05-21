"""
INFO-H3000 - Recherche opérationnelle
Liam Fallik, Quentin Roels

This script sorts an input csv of answers to the problem by distance
"""

import csv
import three_trucks


def make_list_from_csv(path: str):
    """Import the csv as a list of individuals"""
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def sorted_population_dist(pop):
    """Sort the individuals by distance"""
    list_score = []

    for ind in pop:
        list_score.append((ind, three_trucks.find_total_dist(ind), three_trucks.find_weighted_dist(ind)))

    return sorted(list_score, key=lambda x: x[1], reverse=False)


if __name__ == '__main__':
    three_trucks.distance_matrix = three_trucks.import_matrix("../resources/matrix.csv")

    population = make_list_from_csv("../results/To Keep/New/all_data_filtered")
    sorted_pop = sorted_population_dist(population)

    three_trucks.save_csv(sorted_pop, "../results/To Keep/New/filtered_sorted")
