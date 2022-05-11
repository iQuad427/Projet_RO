from three_trucks import *


def simulate_one_weight(weight: float):
    results_ind = []
    results_scores = []

    for tries in range(tries_on_same_weight):
        population = genetic_generate_init()
        weight_constant = calculate_weight_constant(population)
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


if __name__ == '__main__':
    weight = 0.75
    x, y, results = simulate_one_weight(weight)
    file_name = f"results/{pop_size}pop_{number_of_counties}_{number_of_truck}_{it}it_{tries_on_same_weight}try_{weight}weight_{round(children_fraction_in_population * 100)}child.csv"
    save_csv(results, file_name)
