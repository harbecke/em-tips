from collections import defaultdict

import numpy as np
import nashpy as nash


def calculate_cross_probs(effects, result_probs):
    num_results = len(result_probs)
    cross_probs = []
    sum_probs = []

    for idx1 in range(len(effects[0])):
        idx1_list = []
        sum_dict = defaultdict(lambda: [0]*num_results)

        for idx2 in range(len(effects[0])):
            idx2_dict = defaultdict(float)

            for idx3, prob in enumerate(result_probs):
                idx2_dict[effects[idx3%16][idx1] - effects[idx3%16][idx2] +5*(idx3//16)] += prob
                sum_dict[effects[idx3%16][idx1] - effects[idx3%16][idx2] +5*(idx3//16)][idx2] += prob

            idx1_list.append(idx2_dict)

        cross_probs.append(idx1_list)
        sum_probs.append(sum_dict)

    return cross_probs, sum_probs


def calculate_rewards(sum_probs, current_diff, value_dict):
    equation_list = []
    for strat in sum_probs:
        value_sum = np.array([0]*len(sum_probs), dtype=float)
        for key, value in strat.items():
            value_sum += np.array(value[:16])*value_fun(current_diff-key, value_dict)

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

    if verbose:
        print(results_dict)
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
    results_p1 = [0.148, 0.12, 0.109, 0.105, 0.09, 0.077, 0.074, 0.054, 0.054, 0.037, 0.032, 0.029, 0.022, 0.021, 0.019,
                  0.009]
    results_p2 = [0.13, 0.173, 0.13, 0.069, 0.104, 0.038, 0.124, 0.027, 0.030, 0.052, 0.055, 0.011, 0.022, 0.006, 0.022,
                  0.007]
    results_p3 = [0.133, 0.129, 0.131, 0.12, 0.059, 0.069, 0.057, 0.061, 0.025, 0.023, 0.015, 0.026, 0.011, 0.021,
                  0.009, 0.001, 0.015, 0, 0.007, 0, 0.015, 0, 0.015, 0, 0.018, 0.009, 0.011, 0, 0.008, 0, 0.006, 0.006]
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

    value_dict = {0: 1}
    # cross_probs1, sum_probs1 = calculate_cross_probs(effects, results_p1)
    # cross_probs2, sum_probs2 = calculate_cross_probs(effects, results_p2)
    cross_probs3, sum_probs3 = calculate_cross_probs(effects, results_p3)

    # value_dict = calculate_win_chance(sum_probs1, range(-1, 0), value_dict, verbose=True)
    # value_dict = calculate_win_chance(sum_probs1, range(-5, 5), value_dict)
    value_dict = calculate_win_chance(sum_probs3, range(4, 5), value_dict, verbose=True)


if __name__ == "__main__":
    main()
