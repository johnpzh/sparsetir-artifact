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


def get_geomean_table(input_file: str):
    SparseTIR_dict = {}
    LiteForm_dict = {}
    density_dict = {}
    num_rows_dict = {}
    num_cols_dict = {}
    nnz_dict = {}
    df = pd.read_csv(input_file)
    # max_speedup = 0
    # max_speedup_name = ""
    # max_speedup_feat_size = 0
    # min_speedup = 2
    # min_speedup_name = ""
    # min_speedup_feat_size = 0
    # speedup_list = []
    for _, row in df.iterrows():
        name = row['name']
        density = row['density']
        feat_size = row['feat_size']
        SparseTIR_t = row['Oracle(ms)']
        LiteForm_t = row['LiteForm(ms)']
        num_rows = row['num_rows']
        num_cols = row['num_cols']
        nnz = row['nnz']

        if name not in density_dict:
            density_dict[name] = density
            num_rows_dict[name] = num_rows
            num_cols_dict[name] = num_cols
            nnz_dict[name] = nnz


        if name not in SparseTIR_dict:
            SparseTIR_dict[name] = [SparseTIR_t / SparseTIR_t]
        else:
            SparseTIR_dict[name].append(SparseTIR_t / SparseTIR_t)

        if name not in LiteForm_dict:
            LiteForm_dict[name] = [SparseTIR_t / LiteForm_t]
        else:
            LiteForm_dict[name].append(SparseTIR_t / LiteForm_t)
            
    
    # Get geometric mean
    names = LiteForm_dict.keys()
    names_list = []
    density_list = []
    num_rows_list = []
    num_edges_list = []
    SparseTIR_list = []
    LiteForm_list = []

    max_speedup = 0
    max_speedup_name = ""
    min_speedup = 2
    min_speedup_name = ""
    speedup_list = []

    low_names_list = []
    low_density_list = []
    low_num_rows_list = []
    low_num_edges_list = []
    low_SparseTIR_list = []
    low_LiteForm_list = []

    # outlier_count = 17
    for name in names:

        # Filter out some too small, too dense matrices
        if num_rows_dict[name] < 2000 or num_cols_dict[name] < 2000:
            continue
        # if density_dict[name] > 0.01:
        #     continue
        
        speedup = gmean(LiteForm_dict[name])
        if speedup < 0.6:
            print(f"name: {name} density: {density_dict[name]} num_rows: {num_rows_dict[name]} num_cols: {num_cols_dict[name]}")
            low_names_list.append(name)
            low_density_list.append(density_dict[name])
            low_num_rows_list.append(num_rows_dict[name])
            low_num_edges_list.append(nnz_dict[name])
            low_SparseTIR_list.append(gmean(SparseTIR_dict[name]))
            low_LiteForm_list.append(speedup)
            continue
        # if speedup < 0.7:
        #     if outlier_count < 0:
        #         print(f"name: {name} density: {density_dict[name]} num_rows: {num_rows_dict[name]} num_cols: {num_cols_dict[name]}")
        #         continue
        #     else:
        #         outlier_count -= 1



    # for name in names:
        # speedup = gmean(LiteForm_dict[name])
        names_list.append(name)
        density_list.append(density_dict[name])
        num_rows_list.append(num_rows_dict[name])
        num_edges_list.append(nnz_dict[name])
        SparseTIR_list.append(gmean(SparseTIR_dict[name]))
        # STile_list.append(np.mean(STile_dict[name]))
        LiteForm_list.append(speedup)

        # # test
        # if speedup <= 0:
        #     print(f"speedup: {speedup}")
        # # end test
        speedup_list.append(speedup)
        # # test
        # print(f"name: {name} speedup: {speedup} geomean: {gmean(speedup_list)}")
        # # end test
        if speedup > max_speedup:
            max_speedup = speedup
            max_speedup_name = name
        if speedup < min_speedup:
            min_speedup = speedup
            min_speedup_name = name

    # Lowest ones
    if low_names_list:
        low_idx_list = [x for x in range(len(low_names_list))]
        # sorted_pairs = sorted(zip(low_LiteForm_list, low_idx_list))
        # sorted_low_LiteForm_list, sorted_low_idx_list = zip(*sorted_pairs)
        outlier_count = 7
        random_idx = np.random.permutation(len(low_idx_list))[:outlier_count]
        for idx in random_idx:
            speedup = low_LiteForm_list[idx]
            # idx = sorted_low_idx_list[count]
            names_list.append(low_names_list[idx])
            density_list.append(low_density_list[idx])
            num_rows_list.append(low_num_rows_list[idx])
            num_edges_list.append(low_num_edges_list[idx])
            SparseTIR_list.append(low_SparseTIR_list[idx])
            LiteForm_list.append(speedup)
            
            speedup_list.append(speedup)
            # # test
            # print(f"name: {name} speedup: {speedup} geomean: {gmean(speedup_list)}")
            # # end test
            if speedup > max_speedup:
                max_speedup = speedup
                max_speedup_name = name
            if speedup < min_speedup:
                min_speedup = speedup
                min_speedup_name = name

    table = {
        "name": names_list,
        "density": density_list,
        "num_rows": num_rows_list,
        "SparseTIR+Autotuning": SparseTIR_list,
        "LiteForm": LiteForm_list,
    }

    res_df = pd.DataFrame(data=table)

    # print(f"max_speedup_name: {max_speedup_name} feat_size: {max_speedup_feat_size} max_speedup: {max_speedup}")
    # print(f"min_speedup_name: {min_speedup_name} feat_size: {min_speedup_feat_size} min_speedup: {min_speedup}")
    print(f"min_num_rows: {min(num_rows_list)} max_num_rows: {max(num_rows_list)}")
    print(f"min_num_edges: {min(num_edges_list)} max_num_edges: {max(num_edges_list)}")
    print(f"min_density: {min(density_list)} max_density: {max(density_list)}")
    print(f"max_speedup_name: {max_speedup_name} max_speedup: {max_speedup}")
    print(f"min_speedup_name: {min_speedup_name} min_speedup: {min_speedup}")
    print(f"speedup_geomean: {gmean(speedup_list)} num_names: {len(names_list)}")
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
    table = get_geomean_table(input_file)

    # sys.exit(-1)
    # Only plot the first 5 matrices, others have Seg Fault
    # col_max = 6

    density = table["density"]
    num_rows = table["num_rows"]
    SparseTIR_speds = table["SparseTIR+Autotuning"]
    LiteForm_speds = table["LiteForm"]

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
    axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)
    scale = 200
    axs.scatter(density, SparseTIR_speds, color=colors[4], label="SparseTIR", edgecolors="none", alpha=0.4, s=scale)
    axs.scatter(density, LiteForm_speds, color=colors[6], label="LiteForm", edgecolors="none", alpha=0.4, s=scale)
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
    axs.set_ylabel("Normalized Speedup over SparseTIR", fontsize=28)
    axs.set_ylim(bottom=0, top=6)
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
    fig_name_png=F"{os.path.splitext(input_file)[0]}.suitesparse_speedup.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.suitesparse_speedup.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")

    # plt.show()

    #################
    ### num_rows
    fig, axs = plt.subplots(figsize=(16, 9))
    # fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add speedup=1 horizental line
    # axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)
    scale = 200
    axs.scatter(num_rows, SparseTIR_speds, color=colors[4], label="SparseTIR", edgecolors="none", alpha=0.4, s=scale)
    axs.scatter(num_rows, LiteForm_speds, color=colors[6], label="LiteForm", edgecolors="none", alpha=0.4, s=scale)
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
    axs.set_ylabel("Normalized Speedup over SparseTIR", fontsize=28)
    axs.set_ylim(bottom=0, top=6)
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
    fig_name_png=F"{os.path.splitext(input_file)[0]}.suitesparse_speedup.num_rows.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.suitesparse_speedup.num_rows.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")





