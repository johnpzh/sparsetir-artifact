import os
import sys
import argparse
import pandas as pd


OUTPUT_DIR = "output.microbench"

def check_if_valid(name: str,
                   feat_size: int):
    output_dir = OUTPUT_DIR
    filename = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    if not os.path.exists(filename):
        print(F"File {filename} does not exist. Skipped it.")
        return False
    
    return True

def get_names(csv_filename: str):
    df = pd.read_csv(csv_filename)

    names_list = []
    feat_sizes_list = []


    for ind, row in df.iterrows():
        names_list.append(row['name'])
        feat_sizes_list.append(row['K'])

    return (names_list, feat_sizes_list)


def combine(csv_filename):
    output_dir = OUTPUT_DIR
    names_list, feat_sizes_list = get_names(csv_filename)
    output_filename = os.path.join(output_dir, F"output_0_hyb_microbench_collect.csv")

    with open(output_filename, "w") as fout:
        is_first = True
        for name, feat_size in zip(names_list, feat_sizes_list):
            # for feat_size in [32, 64, 128, 256, 512]:
            if not check_if_valid(name, feat_size):
                continue
            input_filename = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
            with open(input_filename, "r") as fin:
                lines = fin.readlines()
                if is_first:
                    is_first = False
                    for i in range(len(lines)):
                        fout.write(lines[i])
                else:
                    for i in range(1, len(lines)):
                        fout.write(lines[i])
    
    # new_csv = pd.read_csv(output_filename)
    # old_csv = pd.read_csv(csv_filename)
    # old_exe_time = list(old_csv["exe_time_hyb"])
    # old_buckets = list(old_csv["best_max_bucket_width"])
    # new_exe_time = list(new_csv["exe_time_searched(ms)"])
    # speedup = [F"{old_time/new_time:.6}" for old_time, new_time in zip(old_exe_time, new_exe_time)]
    # new_csv = new_csv.assign(**{'best_max_bucket_width': old_buckets,
    #                             'exe_time_hyb(ms)': old_exe_time,
    #                             'speedup_to_autotune': speedup})
    # new_csv.to_csv(output_filename, index=False)
                        
    
    print(F"Saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("collect hyb results")
    parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    csv_filename = args.feature_csv
    combine(csv_filename)