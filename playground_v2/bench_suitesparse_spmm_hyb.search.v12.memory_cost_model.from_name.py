# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import os
import sys
import argparse
import torch as th
import math
import pandas as pd
import time
from format_matrix_market import MTX
from format_algos import (
    build_hyb_format,
    bench_hyb_with_config,
    search_bucket_config,
    CostModelSettings
)

OUTPUT_DIR="output.cost_model_name"


# def save_statistics(features: dict,
#                     execution_times: list,
#                     partitions: list,
#                     max_bucket_sizes: list):
def save_statistics(features: dict):
    name = features["name"]
    feat_size = features["K"]

    # Pandas settings
    pd.set_option("display.width", 800)
    pd.set_option("display.max_columns", None)

    dataFrame = pd.DataFrame(data=features)
    dataFrame.set_index("name", inplace=True)
    log = OUTPUT_DIR
    if not os.path.exists(F"{log}"):
        os.mkdir(F"{log}")
    log = os.path.join(log, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    print(dataFrame)
    dataFrame.to_csv(log)

    # # Put prints under output/ directory
    # columns = {
    #     "num_partitions": partitions,
    #     "max_bucket_size": max_bucket_sizes,
    #     "exe_time": execution_times
    # }
    # dataFrame = pd.DataFrame(data=columns)
    # dataFrame.set_index("num_partitions", inplace=True)
    # print(dataFrame)
    # dataFrame.to_csv(log, mode='a')

    print(F"#### Saved to {log} .")



# col_part_config = {
#     "cora": 1,
#     "citeseer": 1,
#     "pubmed": 1,
#     "ppi": 16,
#     "arxiv": 1,
#     "proteins": 8,
#     "reddit": 8,
#     "products": 16,
# }

# bucketing_config = {
#     "cora": [1, 2, 4],
#     "citeseer": [1, 2, 4],
#     "pubmed": [1, 2, 4, 8, 16, 32],
#     "ppi": [1, 2, 4, 8, 16, 32],
#     "arxiv": [1, 2, 4, 8, 16, 32],
#     "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
#     "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
#     "products": [1, 2, 4, 8, 16, 32],
# }

def check_if_done_before(name: str,
                         feat_size: int):
    output_dir = OUTPUT_DIR
    filename = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    if os.path.isfile(filename):
        print(F"{filename} already exits. Skipped it.")
        return True
    else:
        return False
    

def get_names_and_num_partitions(csv_filename: str):
    df = pd.read_csv(csv_filename)

    names_list = []
    num_partitions_list = []
    feat_sizes_list = []
    for ind, row in df.iterrows():
        names_list.append(row['name'])
        num_partitions_list.append(row['best_num_partitions'])
        feat_sizes_list.append(row['K'])
    
    return (names_list, feat_sizes_list, num_partitions_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    parser.add_argument("--num-partitions", "-p", type=int, help="number of column partitions")
    parser.add_argument("--feat-size", "-f", type=int, help="feature size (N) of matrix B")
    # parser.add_argument("--feature-csv", "-f", type=str, help="input feature csv file")
    parser.add_argument("--implicit-unroll", "-i", action="store_true", default=True, help="use implicit unroll")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    filename = args.dataset
    g = MTX(filename)
    num_parts = args.num_partitions
    feat_size = args.feat_size
    name = g.name
    # csv_filename = args.feature_csv

    # names_list, feat_sizes_list, num_partitions_list = get_names_and_num_partitions(csv_filename)

    # GPU device
    print(F"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', default=0)}")

    # cost_model_config = CostModelSettings(mem_w=0.01, bub_w=0.99)
    # cached_config = {}

    # for name, feat_size, num_parts in zip(names_list, feat_sizes_list, num_partitions_list):
    print("")
    print(F"Going to matrix {name} ...")
    # filename = F"/raid/peng599/scratch/spmm/data/suitesparse/{name}/{name}.mtx"
    # g = MTX(filename)

    # if name not in cached_config:
    start_time = time.perf_counter()
    # bucket_config = search_bucket_sizes(g, num_parts)
    cost_model_config = CostModelSettings(feat_size=feat_size, num_parts=num_parts)
    bucket_config = search_bucket_config(g, num_parts, cost_model_config)
    end_time = time.perf_counter()
    search_overhead = end_time - start_time
    # cached_config[name] = (bucket_config, search_overhead)
    # else:
    #     bucket_config, search_overhead = cached_config[name]
    # test
    # bucket_config = [[1, 2, 4, 8, 16, 32, 64]]
    # bucket_config = [[8, 128, 1024]]
    # end test

    # for feat_size in [32]:
    # for feat_size in [32, 64, 128, 256, 512]:
    features = g.matrix_features()
    features["K"] = feat_size

    # # If done before, skipped
    if check_if_done_before(features["name"], feat_size):
        sys.exit(-1)

    # x = th.rand((g.num_dst_nodes(), feat_size))
    x = th.ones((g.num_dst_nodes(), feat_size))
    # y_golden = dgl.ops.copy_u_sum(g, x)
    y_ndarray = g.dot(x.numpy())

    hyb_format = build_hyb_format(g,
                                bucket_config)
    name = features["name"]
    print(F"#### data: {name} feat_size: {feat_size} num_partitions: {num_parts} cost_model_config: {cost_model_config} bucket_config: {bucket_config}", flush=True)
    try:
        exe_time = bench_hyb_with_config(g,
                                x,
                                # y_golden,
                                y_ndarray,
                                feat_size=feat_size,
                                bucket_widths=bucket_config,
                                bucketing_format=hyb_format,
                                coarsening_factor=2,
                                num_col_parts=num_parts,
                                use_implicit_unroll=args.implicit_unroll)
        exe_time = float(F"{exe_time:.6}")
    except Exception as e:
        exe_time = math.inf
        print(e, file=sys.stderr)
    features["best_num_partitions"] = num_parts
    features["bucket_widths"] = [str(bucket_config)]
    features["search_time(s)"] = search_overhead
    features["exe_time_searched(ms)"] = [exe_time]

    save_statistics(features)
