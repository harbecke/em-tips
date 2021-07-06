from collections import defaultdict

import numpy as np
import nashpy as nash


def calculate_cross_probs(effects, result_probs):
    num_results = len(result_probs)
    cross_probs = []
    sum_probs = []

    for idx1 in range(num_results):
        idx1_list = []
        sum_dict = defaultdict(lambda: [0]*num_results)

        for idx2 in range(num_results):
            idx2_dict = defaultdict(float)

            for idx3, prob in enumerate(result_probs):
                idx2_dict[effects[idx3][idx1] - effects[idx3][idx2]] += prob
                sum_dict[effects[idx3][idx1] - effects[idx3][idx2]][idx2] += prob

            idx1_list.append(idx2_dict)

        cross_probs.append(idx1_list)
        sum_probs.append(sum_dict)

    return cross_probs, sum_probs


def calculate_rewards(sum_probs, current_diff, value_dict):
    equation_list = []
    for strat in sum_probs:
        value_sum = np.array([0]*len(sum_probs), dtype=float)
        for key, value in strat.items():
            value_sum += np.array(value)*value_fun(current_diff-key, value_dict)

        equation_list.append(value_sum)

    return np.array(equation_list)


def calculate_win_chance(sum_probs, eval_list, value_dict, verbose=False):
    results_dict = dict()

    for idx in eval_list:
        A = calculate_rewards(sum_probs, idx, value_dict)
        A = np.transpose(A)

        rps = nash.Game(A)
        play_counts = list(rps.vertex_enumeration())
        print("result")

        value_row = np.transpose(A).dot(play_counts[0][0])
        value_col = A.dot(play_counts[0][1])
        if verbose:
            print(A)
            print(play_counts)

        assert(abs(min(value_row) - max(value_col)) < 0.0001)

        results_dict[idx] = min(value_row)

    return results_dict


def value_fun(x, dict):
    if x in dict.keys():
        return dict[x]
    elif x < 0:
        return 0
    else:
        return 1


def main():
    # ["1-1", "1-0", "0-0", "0-1", "2-1", "1-2", "2-0", "0-2", "2-2", "3-1", "3-0", "1-3", "3-2", "0-3", "2-3", "3-3"]
    results_p1 = [0.143, 0.116, 0.105, 0.102, 0.087, 0.074, 0.071, 0.053, 0.052, 0.036, 0.031, 0.028, 0.021, 0.021,
                  0.018, 0.009]
    results_p2 = [0.116, 0.167, 0.106, 0.067, 0.091, 0.042, 0.115, 0.026, 0.037, 0.05, 0.057, 0.012, 0.022, 0.007, 0.01,
                  0.007]
    effects = [
        [5, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4],
        [0, 5, 0, 0, 4, 0, 3, 0, 0, 3, 3, 0, 4, 0, 0, 0],
        [4, 0, 5, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 5, 0, 4, 0, 3, 0, 0, 0, 3, 0, 3, 4, 0],
        [0, 4, 0, 0, 5, 0, 3, 0, 0, 3, 3, 0, 4, 0, 0, 0],
        [0, 0, 0, 4, 0, 5, 0, 3, 0, 0, 0, 3, 0, 3, 4, 0],
        [0, 3, 0, 0, 3, 0, 5, 0, 0, 4, 3, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 3, 0, 5, 0, 0, 0, 4, 0, 3, 3, 0],
        [4, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 4],
        [0, 3, 0, 0, 3, 0, 4, 0, 0, 5, 3, 0, 3, 0, 0, 0],
        [0, 3, 0, 0, 3, 0, 3, 0, 0, 3, 5, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 3, 0, 4, 0, 0, 0, 5, 0, 3, 3, 0],
        [0, 4, 0, 0, 4, 0, 3, 0, 0, 3, 3, 0, 5, 0, 0, 0],
        [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 0, 5, 3, 0],
        [0, 0, 0, 4, 0, 4, 0, 3, 0, 0, 0, 3, 0, 3, 5, 0],
        [4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5]
    ]
    value_dict = {0: 0.5}
    cross_probs1, sum_probs1 = calculate_cross_probs(effects, results_p1)
    cross_probs2, sum_probs2 = calculate_cross_probs(effects, results_p2)

    value_dict = calculate_win_chance(sum_probs1, range(-5, 6), value_dict)
    value_dict = calculate_win_chance(sum_probs2, range(-6, 5), value_dict)
    value_dict = calculate_win_chance(sum_probs1, range(-1, 0), value_dict, verbose=True)
    print(value_dict)


if __name__ == "__main__":
    main()
