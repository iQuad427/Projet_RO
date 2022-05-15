import csv
import one_truck


def make_list_from_csv(path: str):
    with open(path) as file:
        input_data = [list(map(int, rec)) for rec in csv.reader(file)]

    return input_data


def score_solutions_three_trucks(solutions: list):
    scores = []
    for solution in solutions:
        scores.append([solution, one_truck.find_weighted_dist(solution), one_truck.find_total_dist(solution)])

    return scores


def sort_pareto_one_truck(solutions_scores: list):
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
    data = make_list_from_csv("../results/To Keep/One/all_data")
    data, bad_data = sort_pareto_one_truck(score_solutions_three_trucks(data))

    one_truck.save_csv(data, "../results/To Keep/One/all_data_filtered")
    one_truck.save_csv(bad_data, "../results/To Keep/One/filtered_out")
