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
    # train_data_size = table["data_size"]
    # predict_accuracy = table["predict_accuracy"]
    max_bucket_width_list = table["max_bucket_width"]
    cost_list = table["cost"]
    compute_throughput = table["Compute Throughput(%)"]
    exe_time_list = table["time_exe_our(ms)"]

    cost_list = [float(x) / max(cost_list) for x in cost_list]
    compute_throughput = [float(x) / max(compute_throughput) for x in compute_throughput]
    exe_time_list = [float(x) / max(exe_time_list) for x in exe_time_list]
     

    cmap = mpl.colormaps['tab20']
    # cmap = mpl.colormaps['tab20c']
    num_bars = 3
    colors = np.flip(cmap(np.linspace(0, 1, num_bars)), axis=0)

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
    # fig, axs1 = plt.subplots(figsize=(16, 12))
    fig, axs1 = plt.subplots(figsize=(16, 9))
    # fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    axs1.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)
    width = 6
    axs1.plot(max_bucket_width_list, cost_list, label="Cost value", color=colors[0], linewidth=width)
    axs1.plot(max_bucket_width_list, compute_throughput, label="GPU compute throughput (%)", color=colors[1], linewidth=width)
    axs1.plot(max_bucket_width_list, exe_time_list, label="Execution time (ms)", color=colors[2], linewidth=width)
    

    # axs2 = axs1.twinx()
    # axs2.plot(max_bucket_width_list, compute_throughput, label="Compute throughtput (%)", color=colors[1])
    # rects1 = axs.bar(bars1, SparseTIR_t, width=width, label="SparseTIR Autotune", color=colors[0])
    # rects2 = axs.bar(bars2, STile_t, width=width, label="STile Search", color=colors[1])
    # rects3 = axs.bar(bars3, LiteForm_t, width=width, label="LiteForm Train+Infer", color=colors[2])
    # axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
    # axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")

    # Set axis
    axs1.tick_params(direction="in")
    axs1.set_ylabel("Normalized value", fontsize=40)
    axs1.set_ylim(bottom=0.6)
    # axs1.set_xlim(left=0)
    # axs1.tick_params(axis="y", labelcolor=colors[0])
    # axs.set_yscale("log")
    axs1.set_xscale("log", base=2)
    # axs.set_ylim(top=1000000)
    # axs.set_xticks(bars, matrices, fontsize=20, rotation=0, ha="center")
    # axs.set_xticks(bars, matrices, rotation=45, ha="right")
    # axs2.tick_params(direction="in")
    # axs2.set_ylabel("GPU Compute Throughput (%)", fontsize=26)
    # axs2.tick_params(axis="y", labelcolor=colors[1])

    axs1.set_xlabel("Maximum bucket width", fontsize=40)
    axs1.legend(loc='best', fontsize=34, ncol=1)
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
    fig_name_png=F"{os.path.splitext(input_file)[0]}.cost_model.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.cost_model.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")

    # plt.show()





