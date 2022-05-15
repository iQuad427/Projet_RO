import csv
import one_truck


def make_list_from_csv(path: str):
    with open(path) as file:
        data = [list(map(int, rec)) for rec in csv.reader(file)]

    return data


def sorted_population_dist(pop):
    list_score = []

    for ind in pop:
        list_score.append((ind, one_truck.find_total_dist(ind)))

    return sorted(list_score, key=lambda x: x[1], reverse=False)


if __name__ == '__main__':
    population = make_list_from_csv("../results/To Keep/One/all_data_filtered")
    sorted_pop = sorted_population_dist(population)

    one_truck.save_csv(sorted_pop, "results/To Keep/One/filtered_sorted")
