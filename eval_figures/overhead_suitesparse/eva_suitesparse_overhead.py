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


def get_table(input_file: str):
    # SparseTIR_dict = {}
    # LiteForm_dict = {}
    names = []
    density_dict = {}
    num_rows_dict = {}
    autotuning_dict = {}
    time_select_dict = {}
    time_num_parts_dict = {}
    time_buckets_dict = {}
    df = pd.read_csv(input_file)
    for _, row in df.iterrows():
        name = row['name']
        density = row['density']
        num_rows = row['num_rows']
        # feat_size = row['feat_size']
        autotuning = row['autotuning_time(s)']
        time_select = row['time_select(s)']
        time_num_parts = row['time_num_parts(s)']
        time_buckets = row['time_bucket(s)']
        # nnz = row['nnz']

        if num_rows < 2000:
            continue
        
        if name not in names:
            names.append(name)

        if name not in density_dict:
            density_dict[name] = density

        if name not in num_rows_dict:
            num_rows_dict[name] = num_rows

        if name not in autotuning_dict:
            autotuning_dict[name] = [autotuning]
        else:
            autotuning_dict[name].append(autotuning)

        if name not in time_select_dict:
            time_select_dict[name] = [time_select]
        else:
            time_select_dict[name].append(time_select)

        if name not in time_num_parts_dict:
            time_num_parts_dict[name] = [time_num_parts]
        else:
            time_num_parts_dict[name].append(time_num_parts)

        if name not in time_buckets_dict:
            time_buckets_dict[name] = [time_buckets]
        else:
            time_buckets_dict[name].append(time_buckets)
            
    
    # Get mean
    # names = density_dict.keys()
    names_list = []
    density_list = []
    num_rows_list = []
    SparseTIR_cost_list = []
    LiteForm_cost_list = []
    ratio_list = []

    for name in names:
        names_list.append(name)
        density_list.append(density_dict[name])
        num_rows_list.append(num_rows_dict[name])
        SparseTIR_cost = np.mean(autotuning_dict[name])
        SparseTIR_cost_list.append(SparseTIR_cost)
        LiteForm_cost = np.mean(time_select_dict[name]) + np.mean(time_num_parts_dict[name]) + np.mean(time_buckets_dict[name])
        LiteForm_cost_list.append(LiteForm_cost)
        ratio_list.append (SparseTIR_cost / LiteForm_cost)

    table = {
        "density": density_list,
        "num_rows": num_rows_list,
        "SparseTIR_autotuning": SparseTIR_cost_list,
        "LiteForm_inference": LiteForm_cost_list,
    }

    res_df = pd.DataFrame(data=table)

    print(f"ratio_geomean: {gmean(ratio_list)}")
    print(f"ratio_min: {min(ratio_list)}")
    print(f"ratio_max: {max(ratio_list)}")
    print(f"num_names: {len(names_list)}")
    return res_df


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print(F"Uesage: {sys.argv[0]} <input.csv>")
    #     exit()
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
    num_bars = 7
    colors = np.flip(cmap(np.linspace(0, 1, num_bars)), axis=0)


    # Prepare the data
    # input_file = sys.argv[1]
    # input_file = "scripts/tb.runtime.masked_spgemm.LAGraph_COMET.csv"
    # table = pd.read_csv(input_file)
    table = get_table(input_file)

    # sys.exit(-1)
    # Only plot the first 5 matrices, others have Seg Fault
    # col_max = 6

    density = table["density"]
    num_rows = table["num_rows"]
    SparseTIR_autotuning = table["SparseTIR_autotuning"]
    LiteForm_inference = table["LiteForm_inference"]

    # Bar width and locations
    # width = 0.12
    # bars = np.arange(len(matrices))

    # first_bars = [x - width/2 for x in bars]
    # second_bars = [x + width for x in first_bars]

    # # bars1 = bars
    # bars1 = [x - 3 * width for x in bars]
    # bars2 = [x - 2 * width for x in bars]
    # bars3 = [x - 1 * width for x in bars]
    # bars4 = [x - 0 * width for x in bars]
    # bars5 = [x + 1 * width for x in bars]
    # bars6 = [x + 2 * width for x in bars]
    # bars7 = [x + 3 * width for x in bars]

    # Plot the bars
    # fig, axs = plt.subplots(figsize=(16, 12))
    fig, axs = plt.subplots(figsize=(16, 9))
    # fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    # axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)
    scale = 200
    axs.scatter(density, SparseTIR_autotuning, color=colors[4], label="SparseTIR", edgecolors="none", alpha=0.4, s=scale)
    axs.scatter(density, LiteForm_inference, color=colors[6], label="LiteForm", edgecolors="none", alpha=0.4, s=scale)
    # rects1 = axs.bar(bars1, cuSPARSE_speds, width=width, label="cuSPARSE", color=colors[0])
    # rects2 = axs.bar(bars2, Sputnik_speds, width=width, label="Sputnik", color=colors[1])
    # rects3 = axs.bar(bars3, dgSPARSE_speds, width=width, label="dgSPARSE", color=colors[2])
    # rects4 = axs.bar(bars4, TACO_speds, width=width, label="TACO", color=colors[3])
    # rects5 = axs.bar(bars5, SparseTIR_speds, width=width, label="SparseTIR", color=colors[4])
    # rects6 = axs.bar(bars6, STile_speds, width=width, label="STile", color=colors[5])
    # rects7 = axs.bar(bars7, Ours_speds, width=width, label="LiteForm(ours)", color=colors[6])
    # axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
    # axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")

    # Set axis
    axs.tick_params(direction="in")
    axs.set_ylabel("Overhead (s)", fontsize=40)
    # axs.set_ylim(bottom=0, top=6)
    axs.set_yscale("log")
    axs.set_xlabel("Density of matrices", fontsize=40)
    axs.set_xscale("log")
    # axs.set_ylim(bottom=1.0, top=2.3)
    # axs.set_ylim(top=4.5)
    # axs.set_xticks(bars, matrices, fontsize=26, rotation=0, ha="center")
    # axs.set_xticks(bars, matrices, rotation=45, ha="right")
    axs.legend(loc='best', fontsize=40, ncol=1)
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
    # axs.bar_label(rects4, fmt="%0.2f", padding=3, color=colors[3], fontsize=17, rotation=90)
    # axs.bar_label(rects5, fmt="%0.2f", padding=3, color=colors[4], fontsize=17, rotation=90)
    # axs.bar_label(rects6, fmt="%0.2f", padding=3, color=colors[5], fontsize=17, rotation=90)
    # axs.bar_label(rects7, fmt="%0.2f", padding=3, color=colors[6], fontsize=17, rotation=90)
    # axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=12, rotation=45)
    # axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=12, rotation=45)

    # Save the plot
    # baseline = os.path.splitext()
    fig_name_png=F"{os.path.splitext(input_file)[0]}.suitesparse_overhead.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.suitesparse_overhead.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")

    # plt.show()

    ################
    ## Number of rows

    # Plot the bars
    # fig, axs = plt.subplots(figsize=(16, 12))
    fig, axs = plt.subplots(figsize=(16, 9))
    # fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    # axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)
    scale = 200
    axs.scatter(num_rows, SparseTIR_autotuning, color=colors[4], label="SparseTIR", edgecolors="none", alpha=0.4, s=scale)
    axs.scatter(num_rows, LiteForm_inference, color=colors[6], label="LiteForm", edgecolors="none", alpha=0.4, s=scale)
    # rects1 = axs.bar(bars1, cuSPARSE_speds, width=width, label="cuSPARSE", color=colors[0])
    # rects2 = axs.bar(bars2, Sputnik_speds, width=width, label="Sputnik", color=colors[1])
    # rects3 = axs.bar(bars3, dgSPARSE_speds, width=width, label="dgSPARSE", color=colors[2])
    # rects4 = axs.bar(bars4, TACO_speds, width=width, label="TACO", color=colors[3])
    # rects5 = axs.bar(bars5, SparseTIR_speds, width=width, label="SparseTIR", color=colors[4])
    # rects6 = axs.bar(bars6, STile_speds, width=width, label="STile", color=colors[5])
    # rects7 = axs.bar(bars7, Ours_speds, width=width, label="LiteForm(ours)", color=colors[6])
    # axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
    # axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")

    # Set axis
    axs.tick_params(direction="in")
    axs.set_ylabel("Overhead (s)", fontsize=40)
    # axs.set_ylim(bottom=0, top=6)
    axs.set_yscale("log")
    axs.set_xlabel("Number of rows of matrices", fontsize=40)
    axs.set_xscale("log")
    # axs.set_ylim(bottom=1.0, top=2.3)
    # axs.set_ylim(top=4.5)
    # axs.set_xticks(bars, matrices, fontsize=26, rotation=0, ha="center")
    # axs.set_xticks(bars, matrices, rotation=45, ha="right")
    axs.legend(loc='best', fontsize=40, ncol=1)
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
    # axs.bar_label(rects4, fmt="%0.2f", padding=3, color=colors[3], fontsize=17, rotation=90)
    # axs.bar_label(rects5, fmt="%0.2f", padding=3, color=colors[4], fontsize=17, rotation=90)
    # axs.bar_label(rects6, fmt="%0.2f", padding=3, color=colors[5], fontsize=17, rotation=90)
    # axs.bar_label(rects7, fmt="%0.2f", padding=3, color=colors[6], fontsize=17, rotation=90)
    # axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=12, rotation=45)
    # axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=12, rotation=45)

    # Save the plot
    # baseline = os.path.splitext()
    fig_name_png=F"{os.path.splitext(input_file)[0]}.suitesparse_overhead.num_rows.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.suitesparse_overhead.num_rows.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")





