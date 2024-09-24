import numpy as np
import pandas as pd
import argparse
import sys
import os


def drop_duplicate(feature_file: str):
    df = pd.read_csv(feature_file)

    # Drop duplicated matrices by names
    # df = df[df["speed_to_naive"] >= 1.1] # Drop slowdown cases
    # df.drop_duplicates(subset=['name'], inplace=True, keep='last') # Drop duplicate names
    # df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names
    # df.drop(columns=['K'], inplace=True) # Drop column 'K'
    # df.dropna(inplace=True) # Drop rows with any NaN

    num_rows = df.shape[0]
    step = 5
    rows_to_drop = []
    for i in range(0, num_rows, step):
        part = df.iloc[i : i+step]
        if part.isnull().values.any(): # Any of the elements in part is null
            for idx in range(i, i + step):
                rows_to_drop.append(idx)
    df.drop(rows_to_drop, inplace=True)

    # df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names


    # Save
    basename = os.path.splitext(os.path.basename(feature_file))[0]
    output_file = F"{basename}.no_nan.csv"
    df.to_csv(output_file, index=False)
    print(F"Saved to {output_file} .")

    df.drop_duplicates(subset=['name'], inplace=True) # Drop duplicate names

    names = list(df['name'])
    num_parts = 5
    total_count = len(names)
    part_count = (total_count + num_parts - 1) // num_parts
    print(f"total_count: {total_count}")
    for i_p in range(num_parts):
        part_filename = f"output.names.part{i_p}.txt"
        with open(part_filename, "w") as fout:
            i_start = i_p * part_count
            i_bound = (i_p + 1) * part_count
            if i_bound > total_count:
                i_bound = total_count
            print(f"i_p: {i_p} i_start: {i_start} i_bound: {i_bound}")
            for i_n in range(i_start, i_bound):
                fout.write(names[i_n] + "\n")
        print(f"Saved names to {part_filename} ..")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Drop duplicate names in csv")
    parser.add_argument("--features", "-f", type=str, help="features csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    feature_file = args.features
    # split(feature_file, train_ratio)
    drop_duplicate(feature_file)