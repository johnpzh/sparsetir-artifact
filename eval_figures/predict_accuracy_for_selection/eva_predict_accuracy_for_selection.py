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


def get_numbers_for_each_name(input_file: str):
    SparseTIR_dict = {}
    STile_dict = {}
    LiteForm_dict = {}
    df = pd.read_csv(input_file)

    for r_i, row in df.iterrows():
        name = row['name']
        SparseTIR_t = row['SparseTIR+Autotuning']
        STile_t = row['STile']
        LiteForm_t = row['LiteForm(our)']
        if name not in SparseTIR_dict:
            SparseTIR_dict[name] = [SparseTIR_t]
        else:
            SparseTIR_dict[name].append(SparseTIR_t)
        if name not in STile_dict:
            STile_dict[name] = [STile_t]
        else:
            STile_dict[name].append(STile_t)
        if name not in LiteForm_dict:
            LiteForm_dict[name] = [LiteForm_t]
        else:
            LiteForm_dict[name].append(LiteForm_t)
    
    # Get average
    names = LiteForm_dict.keys()
    SparseTIR_list = []
    STile_list = []
    LiteForm_list = []
    for name in names:
        SparseTIR_list.append(np.mean(SparseTIR_dict[name]))
        STile_list.append(np.mean(STile_dict[name]))
        LiteForm_list.append(np.mean(LiteForm_dict[name]))

    table = {
        "name": names,
        "SparseTIR": SparseTIR_list,
        "STile": STile_list,
        "LiteForm": LiteForm_list,
    }

    res_df = pd.DataFrame(data=table)
    return res_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot")
    parser.add_argument("input", type=str, help="data csv file")
    # parser.add_argument("--infer-file", "-i", type=str, help="test csv file for prediction")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    input_file = args.input

    # Fonts and Colors
    plt.rcParams["font.size"] = 40
    # colors = Set2_8.mpl_colors
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # cmap = mpl.colormaps['viridis']
    # cmap = mpl.colormaps['rainbow']
    # cmap = mpl.colormaps['cividis']
    cmap = mpl.colormaps['tab20']
    # cmap = mpl.colormaps['tab20c']
    num_bars = 1
    colors = np.flip(cmap(np.linspace(0, 1, num_bars)), axis=0)


    # Prepare the data
    # table = get_numbers_for_each_name(input_file)
    table = pd.read_csv(input_file)

    # sys.exit(-1)
    # Only plot the first 5 matrices, others have Seg Fault
    # col_max = 6

    # matrices = table["name"]
    # SparseTIR_t = table["SparseTIR"]
    # STile_t = table["STile"]
    # LiteForm_t = table["LiteForm"]
    train_data_size = table["data_size"]
    predict_accuracy = table["predict_accuracy"]

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
    fig, axs = plt.subplots(figsize=(16, 12))
    # fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)

    axs.plot(train_data_size, predict_accuracy, color=colors[0], linewidth=6)
    # rects1 = axs.bar(bars1, SparseTIR_t, width=width, label="SparseTIR Autotune", color=colors[0])
    # rects2 = axs.bar(bars2, STile_t, width=width, label="STile Search", color=colors[1])
    # rects3 = axs.bar(bars3, LiteForm_t, width=width, label="LiteForm Train+Infer", color=colors[2])
    # axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
    # axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")

    # Set axis
    axs.tick_params(direction="in")
    axs.set_ylabel("Prediction accuracy for format selection", fontsize=35)
    axs.set_ylim(bottom=0.7, top=1.0)
    axs.set_xlim(left=0)
    # axs.set_yscale("log")
    # axs.set_ylim(top=1000000)
    # axs.set_xticks(bars, matrices, fontsize=20, rotation=0, ha="center")
    # axs.set_xticks(bars, matrices, rotation=45, ha="right")
    axs.set_xlabel("Training data size (rows)", fontsize=40)
    # axs.legend(loc='best', fontsize=20, ncol=1)
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
    fig_name_png=F"{os.path.splitext(input_file)[0]}.accuracy_selection.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.accuracy_selection.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")

    # plt.show()





