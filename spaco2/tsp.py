import numpy as np


def fitness_func(distance_matrix, x_i):
    """fitness for distance matrix

    args:
        distance_matrix: distance matrix
        x_i: a random sort
    """
    total_distance = 0
    for i in range(1, len(distance_matrix)):
        start_city = x_i[i - 1]
        end_city = x_i[i]
        total_distance += distance_matrix[start_city][end_city]
    total_distance += distance_matrix[x_i[-1]][x_i[0]]

    return total_distance


def get_ss(x_best, x_i, r):
    """calculate the exchange sequence

    args:
        x_best: pbest or gbest
        x_i: the current solution
        r: a random value
    """
    velocity_ss = []
    for i in range(len(x_i)):
        if x_i[i] != x_best[i]:
            j = np.where(x_i == x_best[i])[0][0]
            so = (i, j, r)
            velocity_ss.append(so)
            x_i[i], x_i[j] = x_i[j], x_i[i]

    return velocity_ss


def do_ss(x_i, ss):
    """exchange the sequence

    args:
        x_i: the current solution
        ss: the exchanged sequence
    """
    for i, j, r in ss:
        rand = np.random.random()
        if rand <= r:
            x_i[i], x_i[j] = x_i[j], x_i[i]

    return x_i


def tsp(distance_matrix, iter_max_num=1000):
    """process the tsp steps

    args:
        distance_matrix: the distance matrix
        iter_max_num: the maximum cycle number
    """
    size = 50
    r1 = 0.5  # np.random.rand()
    r2 = 0.6  # np.random.rand()
    fitness_value_lst = []
    city_num = len(distance_matrix)

    pbest_init = np.zeros((size, city_num), dtype=np.int64)
    for i in range(size):
        pbest_init[i] = np.random.choice(
            list(range(city_num)), size=city_num, replace=False
        )

    pbest = pbest_init
    pbest_fitness = np.zeros((size, 1))
    for i in range(size):
        pbest_fitness[i] = fitness_func(distance_matrix, x_i=pbest_init[i])

    gbest = pbest_init[pbest_fitness.argmin()]
    gbest_fitness = pbest_fitness.min()

    fitness_value_lst.append(gbest_fitness)

    for i in range(iter_max_num):
        for j in range(size):
            pbest_i = pbest[j].copy()
            x_i = pbest_init[j].copy()
            ss1 = get_ss(pbest_i, x_i, r1)
            ss2 = get_ss(gbest, x_i, r2)
            ss = ss1 + ss2
            x_i = do_ss(x_i, ss)

            fitness_new = fitness_func(distance_matrix, x_i)
            fitness_old = pbest_fitness[j]
            if fitness_new < fitness_old:
                pbest_fitness[j] = fitness_new
                pbest[j] = x_i

            gbest_fitness_new = pbest_fitness.min()
            gbest_new = pbest[pbest_fitness.argmin()]
            if gbest_fitness_new < gbest_fitness:
                gbest_fitness = gbest_fitness_new
                gbest = gbest_new
            fitness_value_lst.append(gbest_fitness)

    return np.array(gbest), np.array(fitness_value_lst)
