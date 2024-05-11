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



def trim(feature_file: str,
         rows: int):
    df = pd.read_csv(feature_file)

    # # Drop duplicated matrices by names
    # df = df[df["speed_to_naive"] >= 1.1] # Drop slowdown cases
    # # df.drop_duplicates(subset=['name'], inplace=True, keep='last') # Drop duplicate names
    # df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names
    # # df.drop(columns=['K'], inplace=True) # Drop column 'K'
    # # df.dropna(inplace=True) # Drop rows with any NaN
    df.sort_values(['speed_to_naive'], ascending=False, inplace=True)
    if rows > 0:
        df = df.iloc[:rows]

    # Save
    basename = os.path.splitext(os.path.basename(feature_file))[0]
    output_file = F"{basename}.trim_by_speedup.csv"
    df.to_csv(output_file, index=False)
    print(F"Saved to {output_file} .")

    # Names
    output_file = F"{basename}.trim_by_speedup.names_as_bash_array.sh"
    with open(output_file, "w") as fout:
        text = get_names_as_bash_array(df)
        fout.write(text)
    print(F"Saved names to {output_file} as a bash array.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser("trim rows after sorted by speedup descendingly")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    parser.add_argument("--rows", "-r", type=int, default=0, help="number of rows to keep")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    rows = args.rows
    # split(feature_file, train_ratio)
    trim(feature_file, rows)