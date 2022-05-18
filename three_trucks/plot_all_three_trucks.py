"""
INFO-H3000 - Recherche opérationnelle
Liam Fallik, Quentin Roels

This script plots all the solutions stored in a csv file and highlights the best ones (the pareto optimal curve) in red.
"""

import three_trucks
import matplotlib.pyplot as plt
import csv


def make_list_from_csv(path: str):
    """Import the csv as a list of individuals"""
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def to_plot_three_trucks(solutions: list):
    """Return the values to be plotted as the x and y axis of the points"""
    scores = []
    for solution in solutions:
        scores.append([three_trucks.find_weighted_dist(solution), three_trucks.find_total_dist(solution)])

    x_val = [score[1] for score in scores]
    y_val = [score[0] for score in scores]

    return x_val, y_val


def print_all_three_trucks():
    """Plot the points belonging to the pareto optimal curve in red"""
    path = 'results/To Keep/Three/all_data_filtered'
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_three_trucks(sol)
    x_p, y_p = three_trucks.filter_pareto(x_in, y_in)

    plt.plot(x_p, y_p, 'or')


def print_not_optimal():
    """Plot all of the points in another color"""
    path = 'results/To Keep/Three/filtered_out'
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_three_trucks(sol)

    plt.plot(x_in, y_in, 'ob')


if __name__ == '__main__':
    three_trucks.distance_matrix = three_trucks.import_matrix("resources/matrix.csv")

    print_not_optimal()
    print_all_three_trucks()

    plt.xlabel("distance")
    plt.ylabel("risk")
    plt.show()
