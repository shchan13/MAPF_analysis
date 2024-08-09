# -*- coding: UTF-8 -*-
"""
Plot a single figure
"""

from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Parameters for scenarios
    exp_path = "/home/rdaneel/MHCBS/local/hwy_data/"
    map_name = "warehouse-10-20-10-2-1"
    scen_name = "s2s"
    scen_num = 1
    ag_num = 600
    hwy_name = "Human"
    solvers = [
        "EECBS-BP-WDG",
        "EECBS-BP-WDG-HWY",
        "EECBS-FX-BP-WDG-HWY",
        "GPBS",
        "EECBS-FX-BP-WDG",
    ]

    # Parameters for plotting
    font_size = 18
    colors = {
        "EECBS-BP-WDG": "grey",
        "EECBS-BP-WDG-HWY": "deepskyblue",
        "EECBS-FX-BP-WDG-HWY": "red",
        "GPBS": "orange",
        "EECBS-FX-BP-WDG": "purple"
    }

    base_solver:str = "EECBS-BP-WDG"
    y_label:str = "cost"  # "lb"
    out_ylabel = "Cost"
    out_xlabel = "Agent index"

    max_agent_num = 0
    all_paths:Dict[str, Dict[str, Dict[int, Dict]]] = {}
    for solver in solvers:
        path_csv = exp_path + hwy_name + "/" + map_name + "/" + scen_name + "-" + str(scen_num) + \
            "/ag-" + str(ag_num) + "/paths_" + solver + ".csv"
        paths = {"width": 0, "soc": 0, "solb": 0, "paths":{}}
        with open(path_csv, mode='r', encoding='utf-8') as fin:
            print("Open: ", path_csv)
            line = fin.readline().strip().split(',')
            assert line[0] == "width"
            paths["width"] = int(line[1])

            line = fin.readline().strip().split(',')
            assert line[0] == "soc"
            paths["soc"] = int(line[1])

            line = fin.readline().strip().split(',')
            assert line[0] == "solb"
            paths["solb"] = int(line[1])

            for line in fin.readlines():
                line = line.strip().split(',')
                ag_id = int(line[0])
                cost = int(line[1])
                lb = int(line[2])
                path = []
                for t in range(3, len(line)):
                    path.append(int(line[t]))
                assert len(path) == cost + 1
                paths["paths"][ag_id] = {"lb": lb, "cost": cost, "path": path}
        all_paths[solver] = paths
        max_agent_num = max(max_agent_num, len(paths["paths"]))

    # Plot costs
    values:Dict[str,List[int]] = {}
    for (solver, paths) in all_paths.items():
        # if solver == base_solver:
        #     continue
        val = [np.inf for _ in range(max_agent_num)]
        for (ag_id, path) in paths["paths"].items():
            # val[ag_id] = paths["paths"][ag_id][y_label] - \
            #     all_paths[base_solver]["paths"][ag_id][y_label]
            val[ag_id] = paths["paths"][ag_id][y_label]
        values[solver] = val

    fig, axs = plt.subplots(len(values), figsize=(18,6), sharex=True, sharey=True)
    for fidx, (solver, val) in enumerate(values.items()):
        axs[fidx].plot(list(range(len(val))), val, label=solver, color=colors[solver])
        axs[fidx].legend(fontsize=font_size//1.2)
        axs[fidx].grid(axis="y")
        axs[fidx].tick_params(axis='y', labelsize=font_size)
        axs[fidx].set_ylabel(out_ylabel, fontsize=font_size)

    fig.supxlabel(out_xlabel, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.tight_layout()
    plt.show()


    # Plot costs difference with base solver
    values:Dict[str,List[int]] = {}
    for (solver, paths) in all_paths.items():
        if solver == base_solver:
            continue
        val = [np.inf for _ in range(max_agent_num)]
        for (ag_id, path) in paths["paths"].items():
            val[ag_id] = paths["paths"][ag_id][y_label] - \
                all_paths[base_solver]["paths"][ag_id][y_label]
        values[solver] = val

    fig, axs = plt.subplots(len(values), figsize=(18,6), sharex=True, sharey=True)
    for fidx, (solver, val) in enumerate(values.items()):
        axs[fidx].plot(list(range(len(val))), val, label=solver, color=colors[solver])
        axs[fidx].legend(fontsize=font_size//1.2)
        axs[fidx].grid(axis="y")
        axs[fidx].tick_params(axis='y', labelsize=font_size)
        axs[fidx].set_ylabel(out_ylabel, fontsize=font_size)

    fig.supxlabel(out_xlabel, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.tight_layout()
    plt.show()

    # Plot lowerbound
    y_label:str = "lb"
    values:Dict[str,List[int]] = {}
    for (solver, paths) in all_paths.items():
        val = [np.inf for _ in range(max_agent_num)]
        for (ag_id, path) in paths["paths"].items():
            val[ag_id] = paths["paths"][ag_id][y_label]
        values[solver] = val

    fig, axs = plt.subplots(len(values), figsize=(18,6), sharex=True, sharey=True)
    for fidx, (solver, val) in enumerate(values.items()):
        axs[fidx].plot(list(range(len(val))), val, label=solver, color=colors[solver])
        axs[fidx].legend(fontsize=font_size//1.2)
        axs[fidx].grid(axis="y")
        axs[fidx].tick_params(axis='y', labelsize=font_size)
        axs[fidx].set_ylabel(out_ylabel, fontsize=font_size)

    fig.supxlabel(out_xlabel, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
