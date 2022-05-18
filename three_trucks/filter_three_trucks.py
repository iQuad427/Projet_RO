"""
INFO-H3000 - Recherche opérationnelle
Liam Fallik, Quentin Roels

This script filters a csv files containing candidate solutions and keeps only the best ones (the pareto optimal curve)
"""

import csv
import three_trucks


def make_list_from_csv(path: str):
    """Import the csv as a list of individuals"""
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def score_solutions_three_trucks(solutions: list):
    """Append the list of scores to each individual solution"""
    scores = []
    for solution in solutions:
        scores.append([solution, three_trucks.find_weighted_dist(solution), three_trucks.find_total_dist(solution)])

    return scores


def sort_pareto_three_trucks(solutions_scores: list):
    """Separate the solutions"""
    optimal_solutions = []
    not_optimal_solutions = []
    for i in range(len(solutions_scores)):
        opt = True
        for j in range(len(solutions_scores)):
            if solutions_scores[j][1] < solutions_scores[i][1] and solutions_scores[j][2] < solutions_scores[i][2]:
                opt = False
        if opt:
            optimal_solutions.append(solutions_scores[i][0])
        else:
            not_optimal_solutions.append(solutions_scores[i][0])

    return optimal_solutions, not_optimal_solutions


if __name__ == '__main__':
    three_trucks.distance_matrix = three_trucks.import_matrix("../resources/matrix.csv")

    data = make_list_from_csv("../results/To Keep/Three/all_data")
    data, bad_data = sort_pareto_three_trucks(score_solutions_three_trucks(data))

    three_trucks.save_csv(data, "../results/To Keep/Three/all_data_filtered")
    three_trucks.save_csv(bad_data, "../results/To Keep/Three/filtered_out")
