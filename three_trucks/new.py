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


def print_file(path, color):
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_three_trucks(sol)

    plt.plot(x_in, y_in, 'o', color=color)


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


def filter_all_data():
    data = make_list_from_csv("../results/To Keep/New/all_data")

    data, bad_data = sort_pareto_three_trucks(score_solutions_three_trucks(data))

    three_trucks.save_csv(data, "../results/To Keep/New/all_data_filtered")
    three_trucks.save_csv(bad_data, "../results/To Keep/New/filtered_out")


if __name__ == '__main__':
    three_trucks.distance_matrix = three_trucks.import_matrix("../resources/matrix.csv")
    filter_all_data()

    plt.figure(dpi=400)

    print_file("../results/To Keep/New/filtered_out", 'grey')
    print_file("../results/To Keep/New/all_data_filtered", 'red')

    ind = [14, 10, 16, 13, 11, 6, 12, 1, 4, 9, 5, 7, 2, 15, 17, 8, 18, 3, 0, 8, 5, 6]
    plt.plot(three_trucks.find_total_dist(ind), three_trucks.find_weighted_dist(ind), 'o', color='blue')

    plt.show()
