import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from palettable.colorbrewer.qualitative import Set2_8
from scipy.stats import gmean
import math



if __name__ == "__main__":
    parser = argparse.ArgumentParser(f"{sys.argv[0]}")
    parser.add_argument("input_selection", type=str, help="data csv file for Selection")
    parser.add_argument("input_partitions", type=str, help="data csv file for Partitions")
    # parser.add_argument("--infer-file", "-i", type=str, help="test csv file for prediction")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    input_file_selection = args.input_selection
    input_file_partitions = args.input_partitions

    # Fonts and Colors
    plt.rcParams["font.size"] = 40
    # colors = Set2_8.mpl_colors
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # cmap = mpl.colormaps['viridis']
    # cmap = mpl.colormaps['rainbow']
    # cmap = mpl.colormaps['cividis']
    cmap = mpl.colormaps['tab20']
    # cmap = mpl.colormaps['tab20c']
    num_bars = 7
    colors = np.flip(cmap(np.linspace(0, 1, num_bars)), axis=0)


    # Prepare the data
    table_selection = pd.read_csv(input_file_selection)
    table_partitions = pd.read_csv(input_file_partitions)

    # sys.exit(-1)
    # Only plot the first 5 matrices, others have Seg Fault
    # col_max = 6

    # matrices = table["name"]
    # SparseTIR_t = table["SparseTIR"]
    # STile_t = table["STile"]
    # LiteForm_t = table["LiteForm"]
    train_data_size_s = table_selection["data_size"]
    predict_accuracy_s = table_selection["predict_accuracy"]
    train_data_size_p = table_partitions["data_size"]
    predict_accuracy_p = table_partitions["predict_accuracy"]

    # Bar width and locations
    # width = 0.17
    # bars = np.arange(len(matrices))

    # first_bars = [x - width/2 for x in bars]
    # second_bars = [x + width for x in first_bars]

    # bars1 = bars
    # bars1 = [x - 1 * width for x in bars]
    # bars2 = [x - 0 * width for x in bars]
    # bars3 = [x + 1 * width for x in bars]

    # Plot the bars
    # fig, axs = plt.subplots(figsize=(16, 12))
    fig, axs = plt.subplots(figsize=(16, 9))
    # fig, axs = plt.subplots(figsize=(7, 9))
    # fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)

    axs.plot(train_data_size_s, predict_accuracy_s, label="Format Selection", color=colors[4], linewidth=6)
    axs.plot(train_data_size_p, predict_accuracy_p, label="Number of Partitions", color=colors[6], linewidth=6)
    # rects1 = axs.bar(bars1, SparseTIR_t, width=width, label="SparseTIR Autotune", color=colors[0])
    # rects2 = axs.bar(bars2, STile_t, width=width, label="STile Search", color=colors[1])
    # rects3 = axs.bar(bars3, LiteForm_t, width=width, label="LiteForm Train+Infer", color=colors[2])
    # axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
    # axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")

    # Set axis
    axs.tick_params(direction="in")
    axs.set_ylabel("Prediction accuracy", fontsize=40)
    axs.set_ylim(bottom=0.7, top=1.0)
    axs.set_xlim(left=0)
    # axs.set_yscale("log")
    # axs.set_ylim(top=1000000)
    # axs.set_xticks(bars, matrices, fontsize=20, rotation=0, ha="center")
    # axs.set_xticks(bars, matrices, rotation=45, ha="right")
    axs.set_xlabel("Training data size (rows)", fontsize=40)
    axs.legend(loc='best', fontsize=40, ncol=1)
    # axs.legend(loc='upper left', fontsize=20, ncol=1)
    # axs.legend(loc='upper left')
    

    # # test
    # print(F"rects1: {rects1}")
    # for r in rects1:
    #     print(r)
    # # end test
    # Bar label
    # axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=17, rotation=90)
    # axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=17, rotation=90)
    # axs.bar_label(rects3, fmt="%0.2f", padding=3, color=colors[2], fontsize=17, rotation=90)
    # axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=12, rotation=45)
    # axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=12, rotation=45)

    # Save the plot
    # baseline = os.path.splitext()
    fig_name_png=F"{os.path.splitext(sys.argv[0])[0]}.selection_and_partitions.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(sys.argv[0])[0]}.selection_and_partitions.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")

    # plt.show()





