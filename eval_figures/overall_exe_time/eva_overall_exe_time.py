import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from palettable.colorbrewer.qualitative import Set2_8
from scipy.stats import gmean
import math


def insert_speedup(collect: dict,
                   name: str,
                   compare: str,
                   baseline):
    if name not in collect:
        collect[name] = []
    
    if compare in ['', 'OOM', float('nan')]:
        print(F"name: {name} compare: {compare} baseline: {baseline}. Skip because compare is invalid.")
        return
    
    speedup = float(baseline) / float(compare)
    collect[name].append(speedup)

def get_geomean_table(filename: str):
    table = pd.read_csv(filename)
    names = []
    # cuSparseBSR_speedup = {}
    # Triton_speedup = {}
    # Oracle_speedup = {}
    # STile_speedup = {}
    # Ours_speedup = {}
    cuSPARSE_sp = {}
    Triton_sp = {}
    Sputnik_sp = {}
    dgSPARSE_sp = {}
    TACO_sp = {}
    SparseTIR_sp = {}
    STile_sp = {}
    Ours_sp = {}
    for _, row in table.iterrows():
        name = row['name']
        # cuS_CSR = row['cuSPARSE-CSR']
        # cuS_BSR = row['cuSPARSE-BSR']
        cuS = row['cuSPARSE']
        Tri = row['Triton']
        Spu = row['Sputnik']
        # Tri = row['Triton']
        dgS = row['dgSPARSE']
        TACO = row['TACO']
        Spa = row['SparseTIR']
        STi = row['STile']
        # Ora = row['exe_time_hyb(ms)']
        Ours = row['LiteForm(our)']
        # # test
        # print(F"name: {name} cuS_CSR: {cuS_CSR} isnan: {math.isnan(cuS_CSR)}")
        # # end test

        # if math.isnan(cuS_CSR):
        #     print(F"name: {name} cuS_CSR: {cuS_CSR}. Skip")
        #     continue

        if name not in names:
            names.append(name)

        # each Speedup against cuSPARSE-CSR (baseline)
        insert_speedup(cuSPARSE_sp,
                       name=name,
                       compare=cuS,
                       baseline=cuS)
        insert_speedup(Triton_sp,
                       name=name,
                       compare=Tri,
                       baseline=cuS)
        insert_speedup(Sputnik_sp,
                       name=name,
                       compare=Spu,
                       baseline=cuS)
        insert_speedup(dgSPARSE_sp,
                       name=name,
                       compare=dgS,
                       baseline=cuS)
        insert_speedup(TACO_sp,
                       name=name,
                       compare=TACO,
                       baseline=cuS)
        insert_speedup(SparseTIR_sp,
                       name=name,
                       compare=Spa,
                       baseline=cuS)
        insert_speedup(STile_sp,
                       name=name,
                       compare=STi,
                       baseline=cuS)
        insert_speedup(Ours_sp,
                       name=name,
                       compare=Ours,
                       baseline=cuS)
    
    # Generate the table
    # cuS_BSR_sps = []
    # Tri_sps = []
    # STi_sps = []
    # Ora_sps = []
    # Ours_sps = []

    cuSPARSE_speds = []
    Triton_speds = []
    Sputnik_speds = []
    dgSPARSE_speds = []
    TACO_speds = []
    SparseTIR_speds = []
    STile_speds = []
    Ours_speds = []

    for name in names:
        cuSPARSE_speds.append(gmean(cuSPARSE_sp[name]))
        Triton_speds.append(gmean(Triton_sp[name]))
        Sputnik_speds.append(gmean(Sputnik_sp[name]))
        dgSPARSE_speds.append(gmean(dgSPARSE_sp[name]))
        TACO_speds.append(gmean(TACO_sp[name]))
        SparseTIR_speds.append(gmean(SparseTIR_sp[name]))
        STile_speds.append(gmean(STile_sp[name]))
        Ours_speds.append(gmean(Ours_sp[name]))
    
    # Review suggests no geomean in the figure
    # # Add a row of geomean for each benchmark
    # names.append('(geomean)')
    # cuSPARSE_speds.append(gmean([x for x in cuSPARSE_speds if not math.isnan(x)]))
    # Triton_speds.append(gmean([x for x in Triton_speds if not math.isnan(x)]))
    # Sputnik_speds.append(gmean([x for x in Sputnik_speds if not math.isnan(x)]))
    # dgSPARSE_speds.append(gmean([x for x in dgSPARSE_speds if not math.isnan(x)]))
    # TACO_speds.append(gmean([x for x in TACO_speds if not math.isnan(x)]))
    # SparseTIR_speds.append(gmean([x for x in SparseTIR_speds if not math.isnan(x)]))
    # STile_speds.append(gmean([x for x in STile_speds if not math.isnan(x)]))
    # Ours_speds.append(gmean([x for x in Ours_speds if not math.isnan(x)]))

    
    table = {
        'name': names,
        "cuSPARSE":     cuSPARSE_speds,
        "Triton": Triton_speds,
        "Sputnik":     Sputnik_speds,
        "dgSPARSE":     dgSPARSE_speds,
        "TACO":     TACO_speds,
        "SparseTIR":     SparseTIR_speds,
        "STile":     STile_speds,
        "Ours":     Ours_speds,
    }

    df = pd.DataFrame(data=table)
    basename = os.path.splitext(filename)[0]
    # baseline = os.path.splitext(os.path.basename(filename))[0]
    output_filename = F"{basename}.speedup_collect.csv"
    df.to_csv(output_filename, index=False)
    print(df.to_string())

    return df
        

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
    # cmap = mpl.colormaps['tab20']
    # # cmap = mpl.colormaps['tab20c']
    # num_bars = 7
    # colors = np.flip(cmap(np.linspace(0, 1, num_bars)), axis=0)


    # Prepare the data
    # input_file = sys.argv[1]
    # input_file = "scripts/tb.runtime.masked_spgemm.LAGraph_COMET.csv"
    # table = pd.read_csv(input_file)
    table = get_geomean_table(input_file)

    # sys.exit(-1)
    # Only plot the first 5 matrices, others have Seg Fault
    # col_max = 6

    matrices = table["name"]
    # cuSparseBSR_sps = table["cuSPARSE-BSR"]
    # Triton_sps = table["Triton"]
    # STile_sps = table["STile"]
    # Oracle_sps = table["Oracle"]
    # Ours_sps = table["LiteForm(ours)"]
    cuSPARSE_speds = table["cuSPARSE"]
    Triton_speds = table["Triton"]
    Sputnik_speds = table["Sputnik"]
    dgSPARSE_speds = table["dgSPARSE"]
    TACO_speds = table["TACO"]
    SparseTIR_speds = table["SparseTIR"]
    STile_speds = table["STile"]
    Ours_speds = table["Ours"]

    # # LAGraph_runtime = table["LAGraph.SpGEMM-only"]
    # # COMET_runtime = table["COMET.SpGEMM-only"]

    # # eltwise_speedup = []
    # comet_speedup = []
    # for i in range(len(matrices)):
    #     # eltwise_speedup.append(float(LAGraph_eltwise_runtime[i]) / float(COMET_eltwise_runtime[i]))
    #     comet_speedup.append(float(LAGraph_runtime[i]) / float(COMET_runtime[i]))

    # # Add geometric mean
    # a_speedup = np.array(comet_speedup)
    # geomean = a_speedup.prod() ** (1.0 / len(a_speedup))
    # matrices = list(matrices)
    # matrices.append("geomean")
    # comet_speedup.append(geomean)

    cmap = mpl.colormaps['tab20']
    num_bars = 8  # the number of benchmarks
    colors = np.flip(cmap(np.linspace(0, 1, num_bars)), axis=0)

    # Bar width and locations
    width = 0.11
    bars = np.arange(len(matrices))

    # first_bars = [x - width/2 for x in bars]
    # second_bars = [x + width for x in first_bars]

    # bars1 = bars
    bars1 = [x - 3 * width for x in bars]
    bars2 = [x - 2 * width for x in bars]
    bars3 = [x - 1 * width for x in bars]
    bars4 = [x - 0 * width for x in bars]
    bars5 = [x + 1 * width for x in bars]
    bars6 = [x + 2 * width for x in bars]
    bars7 = [x + 3 * width for x in bars]

    # bars1 = [x - 3.5 * width for x in bars]
    # bars2 = [x - 2.5 * width for x in bars]
    # bars3 = [x - 1.5 * width for x in bars]
    # bars4 = [x - 0.5 * width for x in bars]
    # bars5 = [x + 0.5 * width for x in bars]
    # bars6 = [x + 1.5 * width for x in bars]
    # bars7 = [x + 2.5 * width for x in bars]
    # bars8 = [x + 3.5 * width for x in bars]

    # Plot the bars
    # fig, axs = plt.subplots(figsize=(16, 9))
    fig, axs = plt.subplots(figsize=(32, 9))

    # Y grid only
    axs.grid(visible=True, color='grey', axis='y', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add speedup=1 horizental line
    axs.axhline(y=1, color='black', linestyle="dashed", alpha=0.4)

    # rects1 = axs.bar(bars1, cuSPARSE_speds, width=width, label="cuSPARSE", color=colors[0])
    rects1 = axs.bar(bars1, Triton_speds, width=width, label="Triton", color=colors[1])
    rects2 = axs.bar(bars2, Sputnik_speds, width=width, label="Sputnik", color=colors[2])
    rects3 = axs.bar(bars3, dgSPARSE_speds, width=width, label="dgSPARSE", color=colors[3])
    rects4 = axs.bar(bars4, TACO_speds, width=width, label="TACO", color=colors[4])
    rects5 = axs.bar(bars5, SparseTIR_speds, width=width, label="SparseTIR", color=colors[5])
    rects6 = axs.bar(bars6, STile_speds, width=width, label="STile", color=colors[6])
    rects7 = axs.bar(bars7, Ours_speds, width=width, label="LiteForm(ours)", color=colors[7])

    # rects1 = axs.bar(bars1, cuSPARSE_speds, width=width, label="cuSPARSE", color=colors[0])
    # rects2 = axs.bar(bars2, Triton_speds, width=width, label="Triton", color=colors[1])
    # rects3 = axs.bar(bars3, Sputnik_speds, width=width, label="Sputnik", color=colors[2])
    # rects4 = axs.bar(bars4, dgSPARSE_speds, width=width, label="dgSPARSE", color=colors[3])
    # rects5 = axs.bar(bars5, TACO_speds, width=width, label="TACO", color=colors[4])
    # rects6 = axs.bar(bars6, SparseTIR_speds, width=width, label="SparseTIR", color=colors[5])
    # rects7 = axs.bar(bars7, STile_speds, width=width, label="STile", color=colors[6])
    # rects8 = axs.bar(bars8, Ours_speds, width=width, label="LiteForm(ours)", color=colors[7])

    # axs.bar(first_bars, LAGraph_runtime, width=width, label="LAGraph", color=colors[0], hatch="/", edgecolor="white")
    # axs.bar(second_bars, COMET_runtime, width=width, label="COMET", color=colors[1], hatch="-", edgecolor="white")




    # Set axis
    axs.tick_params(direction="in")
    axs.set_ylabel("Normalized Speedup over cuSPARSE", fontsize=27)
    # axs.set_ylim(bottom=1.0, top=2.3)
    # axs.set_ylim(top=4.5)
    axs.set_ylim([0.0625, 20])

    # Set y-axis to log scale
    axs.set_yscale('log', base=2)
    # Use ScalarFormatter to show decimal numbers instead of exponential notation
    axs.yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs.ticklabel_format(axis='y', style='plain')
    # Optional: Ensure no minor ticks if you only want major ticks
    axs.yaxis.set_minor_locator(plt.NullLocator())
    axs.set_yticks([0.0625, 0.25, 1, 4, 16])

    # Set x-axis
    axs.set_xticks(bars, matrices, fontsize=40, rotation=0, ha="center")
    
    # axs.set_xticks(bars, matrices, rotation=45, ha="right")
    # axs.legend(loc='best', fontsize=32, ncol=4)
    axs.legend(loc='upper right', fontsize=32, ncol=4)
    # axs.legend(loc='upper left')

    # # test
    # print(F"rects1: {rects1}")
    # for r in rects1:
    #     print(r)
    # # end test
    # Bar label
    fontsize = 29
    axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[1], fontsize=fontsize, rotation=90)
    axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[2], fontsize=fontsize, rotation=90)
    axs.bar_label(rects3, fmt="%0.2f", padding=3, color=colors[3], fontsize=fontsize, rotation=90)
    axs.bar_label(rects4, fmt="%0.2f", padding=3, color=colors[4], fontsize=fontsize, rotation=90)
    axs.bar_label(rects5, fmt="%0.2f", padding=3, color=colors[5], fontsize=fontsize, rotation=90)
    axs.bar_label(rects6, fmt="%0.2f", padding=3, color=colors[6], fontsize=fontsize, rotation=90)
    axs.bar_label(rects7, fmt="%0.2f", padding=3, color=colors[7], fontsize=fontsize, rotation=90)
    # axs.bar_label(rects8, fmt="%0.2f", padding=3, color=colors[7], fontsize=fontsize, rotation=90)

    # axs.bar_label(rects1, fmt="%0.2f", padding=3, color=colors[0], fontsize=12, rotation=45)
    # axs.bar_label(rects2, fmt="%0.2f", padding=3, color=colors[1], fontsize=12, rotation=45)

    # OOM Text
    axs.text(bars1[5], 0.07, "OOM", color=colors[1], rotation=90, horizontalalignment='center', fontsize=fontsize, weight='bold')
    axs.text(bars1[6], 0.07, "OOM", color=colors[1], rotation=90, horizontalalignment='center', fontsize=fontsize, weight='bold')
    # axs.text(bars2[5], 0.1, "OOM", color=colors[1], rotation=90, horizontalalignment='center', fontsize=fontsize)
    # axs.text(bars2[6], 0.1, "OOM", color=colors[1], rotation=90, horizontalalignment='center', fontsize=fontsize)

    # Save the plot
    # baseline = os.path.splitext()
    fig_name_png=F"{os.path.splitext(input_file)[0]}.speedup_collect.png"
    plt.savefig(fig_name_png, dpi=300, bbox_inches="tight")

    fig_name_pdf=F"{os.path.splitext(input_file)[0]}.speedup_collect.pdf"
    plt.savefig(fig_name_pdf, dpi=300, bbox_inches="tight")

    # plt.show()





