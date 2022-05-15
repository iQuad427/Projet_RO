import one_truck
import matplotlib.pyplot as plt
import csv


def make_list_from_csv(path: str):
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def to_plot_one_truck(solutions: list):
    scores = []
    for solution in solutions:
        scores.append([one_truck.find_weighted_dist(solution), one_truck.find_total_dist(solution)])

    x_val = [score[1] for score in scores]
    y_val = [score[0] for score in scores]

    return x_val, y_val


def score_solutions_one_truck(solutions: list):
    scores = []
    for solution in solutions:
        scores.append([solution, one_truck.find_weighted_dist(solution), one_truck.find_total_dist(solution)])

    return scores


def print_all_one_truck():
    path = '../results/To Keep/One/all_data_filtered'
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_one_truck(sol)
    x_p, y_p = one_truck.filter_value(x_in, y_in)

    plt.plot(x_p, y_p, 'or')


def print_not_optimal():
    path = '../results/To Keep/One/filtered_out'
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_one_truck(sol)

    plt.plot(x_in, y_in, 'ob')


if __name__ == '__main__':
    print_not_optimal()
    print_all_one_truck()

    plt.xlabel("distance")
    plt.ylabel("risque")
    plt.show()


