import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.ensemble import RandomForestClassifier


def get_names_as_bash_array(df) -> str:
    # names = []
    text = "MATRICES=( \\\n"
    for ind, row in df.iterrows():
        # names.append(row['name'])
        name = row['name']
        text += F"\"{name}\" \\\n"
    text += ")\n"

    return text


CHOSEN_ONES = [
    "web-Stanford",
    "TSOPF_RS_b39_c19",
    "wikipedia-20051105",
    "rajat21",
    "case39",
    "circuit_4",
    "trans4",
    "dc1",
    "bcsstk03",
    "thermal1",
    "bcsstm03",
    "rajat15",
    "lowThrust_5",
    "bloweya",
    "std1_Jac3",
    "std1_Jac2_db",
    "ca-HepPh",
    "av41092",
    "Zd_Jac3_db",
    "Zd_Jac6",
    "ca-GrQc",
    "bcsstm13",
    "tols340",
    "a2nnsnsl"
]
# CHOSEN_ONES = [
#     "web-BerkStan",
#     "boyd1",
#     "cbuckle",
#     "uni_chimera_i5",
#     "goodwin",
#     "graham1",
#     "plbuckle",
#     "cz2548",
#     "ex25",
#     "Plants_10NN",
#     "spaceStation_2",
#     "iris_dataset_30NN",
#     "netscience",
#     "spaceStation_3",
#     "1138_bus",
#     "str_600",
#     "M20PI_n",
#     "mesh3em5",
#     "mesh3e1",
#     "rajat14",
#     "odepa400",
#     "spaceStation_1",
#     "impcol_a",
#     "impcol_b",
#     "d_dyn1",
#     "cage3",
#     "b1_ss"
# ]


def trim(feature_file: str):
    df = pd.read_csv(feature_file)

    df = df[df["name"].isin(CHOSEN_ONES)]

    # # # Drop duplicated matrices by names
    # # df = df[df["speed_to_naive"] >= 1.1] # Drop slowdown cases
    # # # df.drop_duplicates(subset=['name'], inplace=True, keep='last') # Drop duplicate names
    # # df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names
    # # # df.drop(columns=['K'], inplace=True) # Drop column 'K'
    # # # df.dropna(inplace=True) # Drop rows with any NaN
    # df.sort_values(['speed_to_naive'], ascending=False, inplace=True)
    # if rows > 0:
    #     df = df.iloc[:rows]

    # Save
    basename = os.path.splitext(os.path.basename(feature_file))[0]
    output_file = F"{basename}.trim_by_names.csv"
    df.to_csv(output_file, index=False)
    print(F"Saved to {output_file} .")

    # # Names
    # output_file = F"{basename}.trim_by_speedup.names_as_bash_array.sh"
    # with open(output_file, "w") as fout:
    #     text = get_names_as_bash_array(df)
    #     fout.write(text)
    # print(F"Saved names to {output_file} as a bash array.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("trim rows and only keep chosen names")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    # parser.add_argument("--rows", "-r", type=int, default=0, help="number of rows to keep")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    # rows = args.rows
    # split(feature_file, train_ratio)
    trim(feature_file)