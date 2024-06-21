import sys
import os
import argparse
import secrets
import copy
from scipy.io import mmread
import time


# def if_all_zeros(count: dict):
#     for val in count.values():
#         if val != 0:
#             return False
    
#     return True


def create_a_row(row_ind: int,
                 degree: int,
                 num_col_limit: int):
    is_visited = [False] * num_col_limit
    row = []
    count = 0
    while count < degree:
        col_ind = secrets.randbelow(num_col_limit)
        if is_visited[col_ind]:
            continue
        is_visited[col_ind] = True
        row.append([row_ind, col_ind])
        count += 1
    
    return row


def synthesize(output_filename: str):
    HEAD_LINE = r"%%MatrixMarket matrix coordinate real general"
    # matrix row configurations
    # {bocket_width: num_rows}
    # mtx_config = {
    #     1: 1025,
    #     2: 513,
    #     4: 257,
    #     8: 129,
    #     16: 65,
    #     32: 33,
    #     64: 17,
    #     128: 9,
    #     256: 5,
    #     512: 3,
    #     1024: 2
    # }
    mtx_config = {
        1: 102401,
        2: 51201,
        4: 25601,
        8: 12801,
        16: 6401,
        32: 3201,
        64: 1601,
        128: 801,
        256: 401,
        512: 201,
        1024: 101
    }
    # mtx_config = {
    #     1: 3,
    #     2: 2,
    #     4: 2,
    # }

    sum_rows = sum(mtx_config.values())
    num_rows = sum_rows * 2
    num_cols = num_rows
    nnz = 0
    for width, rows in mtx_config.items():
        nnz += width * rows
    print(F"num_rows: {num_rows} nnz: {nnz}")

    with open(output_filename, "w") as fout:
        fout.write(HEAD_LINE + "\n")
        fout.write(F"{num_rows} {num_rows} {nnz}\n")
        counts = copy.deepcopy(mtx_config)
        widths = list(counts.keys())

        count_rows = 0
        is_visited = [False] * num_rows
        # row_ind = 0
        while count_rows < sum_rows:
            print(F"Generating row {count_rows + 1}/{sum_rows} ...", end="")
            if count_rows + 1 == sum_rows:
                print("", flush=True)
            else:
                print("", end="\r", flush=True)

            # Get the row index
            row_ind = secrets.randbelow(num_rows)
            # if is_visited[row_ind]:
            #     continue
            # is_visited[row_ind] = True
            while is_visited[row_ind]:
                row_ind = (row_ind + 1) % num_rows
            is_visited[row_ind] = True

            # Select the bucket
            w_curr = secrets.choice(widths)
            # if 0 == counts[w_curr]:
            #     continue
            # counts[w_curr] -= 1
            while 0 == counts[w_curr]:
                w_curr = (2 * w_curr) 
                if w_curr > max(counts.keys()):
                    w_curr = 1
            counts[w_curr] -= 1
                

            # Create the row with the width
            edges = create_a_row(row_ind=row_ind, degree=w_curr, num_col_limit=num_cols)
            for src, dst in edges:
                fout.write(F"{src + 1} {dst + 1} 17\n")

            # row_ind += 1
            count_rows += 1

    print(F"Saved to {output_filename}")

    # test
    mtx = mmread(output_filename)
    print(F"num_rows: {mtx.shape[0]} num_cols: {mtx.shape[1]} nnz: {mtx.nnz}")
    # print(mtx.toarray())
    # end test


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Synthesize matrix for profiling the cost model")
    # parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    # parser.add_argument("--num-partitions", "-p", type=int, help="number of column partitions")
    parser.add_argument("--output-mtx", "-o", type=str, help="output matrix market (mtx) matrix")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    # filename = args.dataset
    # g = MTX(filename)
    # num_parts = args.num_partitions
    output_filename = args.output_mtx
    
    start_time = time.perf_counter()
    synthesize(output_filename)
    end_time = time.perf_counter()

    print(F"Exe_time(s): {end_time - start_time:.6}")