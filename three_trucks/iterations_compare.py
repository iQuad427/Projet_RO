from three_trucks import *
import three_trucks

if __name__ == '__main__':
    """ Has the objective to show the diminution of score over iterations, letting us
    decide how many iterations are optimal to reduce overall computing time """

    three_trucks.distance_matrix = import_matrix("../resources/matrix.csv")

    for x in range(5):
        results_ind = []
        results_scores = []
        x_val = []
        y_val = []

        weight = 0.5
        population = genetic_generate_init()
        weight_cst = calculate_weight_constant()

        for i in range(it):
            sorted_list = sorted_population_score(population, weight, weight_cst)
            population = breed(sorted_list, how_many_parents(amount_of_children))
            res = sorted_population_score(population, weight, weight_cst)
            results_ind.append(res[0][0])

            x_val.append(i)
            y_val.append(score(results_ind[-1], weight, weight_cst))

            results_scores.append(score(results_ind[-1], weight, weight_cst))
            print(results_scores[-1])

        print("Weighted distance :  " + str(find_weighted_dist(results_ind[-1])))
        print("Total distance :     " + str(find_total_dist(results_ind[-1])))
        print(results_ind[-1])
        plt.plot(x_val, y_val)

    plt.ylabel("score")
    plt.xlabel("iterations")
    plt.figtext(.7, .81, f"pop_size = {pop_size}")
    plt.figtext(.7, .75, f"children = {children_fraction_in_population*100}%")
    plt.show()
