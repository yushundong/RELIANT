import logging
import numpy as np
import itertools
from tabulate import tabulate


def tabulate_results(results_dict):
    # Average for different seeds
    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.values()) for res in results_dict[variant]])
        # print(results)
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist(),
                    ),
                )
            )
        )
    return tab_data


def output_results(results_dict, tablefmt="github"):
    variant = list(results_dict.keys())[0]
    col_names = ["Variant"] + list(results_dict[variant][-1].keys())
    tab_data = tabulate_results(results_dict)
    print(tabulate(tab_data, headers=col_names, tablefmt=tablefmt))
