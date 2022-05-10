import one_truck
import three_trucks
import matplotlib.pyplot as plt
import csv


def make_list_from_csv(path: str):
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def to_plot_three_trucks(solutions: list):
    scores = []
    for solution in solutions:
        scores.append([three_trucks.find_weighted_dist(solution), three_trucks.find_total_dist(solution)])

    x_val = [score[1] for score in scores]
    y_val = [score[0] for score in scores]

    return x_val, y_val


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


def score_solutions_three_trucks(solutions: list):
    scores = []
    for solution in solutions:
        scores.append([solution, three_trucks.find_weighted_dist(solution), three_trucks.find_total_dist(solution)])

    return scores


def print_all_three_trucks():
    path = 'results/To Keep/Three Trucks/all_data_filtered'
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_three_trucks(sol)
    x_p, y_p = three_trucks.filter_pareto(x_in, y_in)

    plt.plot(x_p, y_p, 'or')
    plt.show()


def print_all_one_truck():
    path = 'results/To Keep/One Truck/20000pop_19_11it_3try_20prec_75child_2.csv'
    sol = make_list_from_csv(path)

    x_in, y_in = to_plot_one_truck(sol)
    x_p, y_p = one_truck.filter_value(x_in, y_in)

    plt.plot(x_p, y_p, 'or')
    plt.show()


def sort_pareto_one_truck():
    pass


def sort_pareto_three_trucks(solutions_scores: list):
    optimal_solutions = []
    for i in range(len(solutions_scores)):
        opt = True
        for j in range(len(solutions_scores)):
            if solutions_scores[j][1] > solutions_scores[i][1] and solutions_scores[j][2] > solutions_scores[i][2]:
                opt = False
        if opt:
            optimal_solutions.append(solutions_scores[i][0])

    return optimal_solutions


def keep_best_in_csv(solutions: list):
    pass


if __name__ == '__main__':
    print_all_three_trucks()
    # print_all_one_truck()

    # data = make_list_from_csv("results/To Keep/Three Trucks/all_data")
    # data = sort_pareto_three_trucks(score_solutions_three_trucks(data))
    # print(data)
    # three_trucks.save_csv(data, "results/To Keep/Three Trucks/all_data_filtered")
