from one_truck import *


if __name__ == '__main__':
    """ Has the objective to show the diminution of score over iterations, letting us
    decide how many iterations are optimal to reduce overall computing time """

    distance_matrix = import_matrix("resources/matrix.csv")
    results_ind = []
    results_scores = []
    x_val = []
    y_val = []

    weight = 1
    population = genetic_generate_init()

    for i in range(it):
        sorted = sorted_population_score(population, weight)
        population = breed(sorted)
        res = sorted_population_score(population, weight)
        results_ind.append(res[0][0])

        x_val.append(i)
        y_val.append(score(results_ind[-1], weight))

        results_scores.append(score(results_ind[-1], weight))
        print(results_scores[-1])

    print(results_ind[-1])
    plt.plot(x_val, y_val, '-r')
    plt.show()

