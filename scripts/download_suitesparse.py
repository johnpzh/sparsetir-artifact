import os
import sys
import argparse
import ssgetpy
import requests
import time

# https://sparse.tamu.edu/files/ssstats.csv


def download_list():
    url = "https://sparse.tamu.edu/files/ssstats.csv"
    file_name = "ssstats.csv"
    response = requests.get(url)
    with open(file_name, "wb") as fout:
        fout.write(response.content)
        print(F"Downloaded {file_name}")

    return file_name


def get_total_num_matrices(file_name: str):
    with open(file_name) as fin:
        line = fin.readline()
        # number of matrices is the first line
        number = int(line)
        line = fin.readline()
        print(F"Totally {number} matrices. Updated on {line}.")
        return number


def download_all(output_dir: str,
                 num_matrices: int):
    """Download all matrices from SuiteSparse

    Args:
        output_dir (str): save target directory
        num_matrices (int): number of matrices up to now
    """

    # Matrix ID is starting from 1, not 0
    for mtx_ind in range(1, num_matrices + 1):
        result = ssgetpy.search(mtx_ind)
        # Download the matrix into MM format under output_dir and extract it.
        # Reference: https://github.com/drdarshan/ssgetpy/blob/master/demo.ipynb
        print(F"\nDownloading matrix [{mtx_ind}] {result} under {output_dir} ...")
        result.download(format="MM", destpath=output_dir, extract=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download all SuiteSparse matrices")
    parser.add_argument("--dir", "-d", type=str, help="download target directory")
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    output_dir = args.dir

    file_name = download_list()
    num_matrices = get_total_num_matrices(file_name)

    start_time = time.perf_counter()
    download_all(output_dir, num_matrices)
    end_time = time.perf_counter()

    print(F"Downloaded {num_matrices} matrices in total.")
    print(F"Execution_time(s): {end_time - start_time}")
