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
import format_algos


OUTPUT_DIR="output.cost_model_profile"


# def save_statistics(features: dict,
#                     execution_times: list,
#                     partitions: list,
#                     max_bucket_sizes: list):
# def save_statistics(features: dict,
#                     cost_model_configs_list: list,
#                     bucket_configs_list: list,
#                     search_overhead_list: list,
#                     exe_time_list: list):
def save_statistics(features: dict):
    name = features["name"]
    feat_size = features["K"]

    # rows = len(exe_time_list)

    # for key, val in features.items():
    #     features[key] = [val] * rows
    # features["cost_model_config"] = cost_model_configs_list
    # features["exe_time_model(ms)"] = exe_time_list
    # features["bucket_config"] = bucket_configs_list
    # features["search_overhead(s)"] = search_overhead_list

    # # test
    # for key, val in features.items():
    #     print(F"key: {key} len(val): {len(val)} val: {val}")
    # # end test

    # Pandas settings
    pd.set_option("display.width", 800)
    pd.set_option("display.max_columns", None)

    dataFrame = pd.DataFrame(data=features)
    # dataFrame.set_index("name", inplace=True)
    print(dataFrame)
    # log = OUTPUT_DIR
    # if not os.path.exists(F"{log}"):
    #     os.mkdir(F"{log}")
    # log = os.path.join(log, F"output_tune_{name}_feat{feat_size}_hyb.csv")
    # dataFrame.to_csv(log, index=False)

    # print(F"#### Saved to {log} .")

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

# def check_if_done_before(name: str,
#                          feat_size: int):
#     output_dir = OUTPUT_DIR
#     filename = os.path.join(output_dir, F"output_tune_{name}_feat{feat_size}_hyb.csv")
#     if os.path.isfile(filename):
#         print(F"{filename} already exits. Skipped it.", flush=True)
#         return True
#     else:
#         return False
    

# def get_names_and_num_partitions(csv_filename: str):
#     df = pd.read_csv(csv_filename)

#     names_list = []
#     num_partitions_list = []
#     feat_sizes_list = []
#     max_bucket_width_list = []
#     exe_time_list = []
#     for ind, row in df.iterrows():
#         names_list.append(row['name'])
#         num_partitions_list.append(row['best_num_partitions'])
#         feat_sizes_list.append(row['K'])
#         max_bucket_width_list.append(row['best_max_bucket_width'])
#         exe_time_list.append(row['exe_time_hyb'])

#     return (names_list, feat_sizes_list, num_partitions_list, max_bucket_width_list, exe_time_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    # parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    # parser.add_argument("--num-partitions", "-p", type=int, help="number of column partitions")
    parser.add_argument("--input-mtx", "-f", type=str, help="input matrix")
    parser.add_argument("--new-max-width", "-w", type=int, help="new max bucket width")
    parser.add_argument("--implicit-unroll", "-i", action="store_true", default=True, help="use implicit unroll")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()
    # filename = args.dataset
    # g = MTX(filename)
    # num_parts = args.num_partitions
    mtx_filename = args.input_mtx
    new_max_width = args.new_max_width
    g = MTX(mtx_filename)
    features = g.matrix_features()
    feat_size = 32
    features["K"] = feat_size
    x = th.ones((g.num_dst_nodes(), feat_size))
    # y_golden = dgl.ops.copy_u_sum(g, x)
    y_ndarray = g.dot(x.numpy())

    init_bucket_config = [[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]]
    # init_bucket_config = [[1, 2, 4]]
    # init_bucket_config = [[16]]
    # init_bucket_config = [[16, 256]]
    name = features["name"]
    num_parts = len(init_bucket_config)

    # Move the bucket if needed
    init_buckets = format_algos.get_init_buckets(g, num_parts)
    # test
    print(F"\nold buckets...")
    for part_ind in range(num_parts):
        b_pool = init_buckets[part_ind]
        for width, bucket in b_pool.items():
            print(F"part_ind: {part_ind} width: {width} Bucket:{bucket}")
    print(F"old bucket_widths: {format_algos.get_bucket_widths_list(init_buckets[0])}")
    # end test

    memory_overhead = 0
    assert init_bucket_config[0] == format_algos.get_bucket_widths_list(init_buckets[0]), "Bucket config not consistent"
    old_max_bucket_width = init_bucket_config[0][-1]
    if old_max_bucket_width != new_max_width:
        old_w = old_max_bucket_width
        while old_w != new_max_width:
            new_w = old_w // 2
            memory_overhead += format_algos.get_memory_overhead(init_buckets[0], old_w, new_w)
            format_algos.move_largest_bucket(init_buckets[0], old_w, new_w)
            old_w = new_w

    # test
    print(F"\nnew buckets...")
    for part_ind in range(num_parts):
        b_pool = init_buckets[part_ind]
        for width, bucket in b_pool.items():
            print(F"part_ind: {part_ind} width: {width} Bucket:{bucket}")
    print(F"new bucket_widths: {format_algos.get_bucket_widths_list(init_buckets[0])}")
    # end test

    # Some features
    bubble_overhead = format_algos.get_bubble_overhead(init_buckets[0])
    bucket_stats = format_algos.bucket_statistics(init_buckets[0])
    
    
    final_bucket_config = []
    final_bucket_config.append(format_algos.get_bucket_widths_list(init_buckets[0]))
    # # test 
    # final_bucket_config = init_bucket_config
    # # end test
    hyb_format = build_hyb_format(g, final_bucket_config)
    # End move

    # # test
    # final_bucket_config = init_bucket_config
    # hyb_format = build_hyb_format(g, final_bucket_config)
    # # end test

    try:
        # cost_model_config = CostModelSettings(mem_w=mem_w, bub_w=bub_w)
        # # if name not in cached_config:
        # start_time = time.perf_counter()
        # # bucket_config = search_bucket_sizes(g, num_parts)
        # bucket_config = search_bucket_config(g, num_parts, cost_model_config)
        # end_time = time.perf_counter()
        # search_overhead = end_time - start_time
        # # cached_config[name] = (bucket_config, search_overhead)
        # # else:
        # #     bucket_config, search_overhead = cached_config[name]
        # # test
        # # bucket_config = [[1, 2, 4, 8, 16, 32, 64]]
        # # bucket_config = [[8, 128, 1024]]
        # # end test
        # # x = th.rand((g.num_dst_nodes(), feat_size))


        # hyb_format = build_hyb_format(g,
        #                             bucket_config)
        
        print(F"#### data: {name} feat_size: {feat_size} num_partitions: {num_parts} bucket_config: {final_bucket_config}", flush=True)

        exe_time = bench_hyb_with_config(g,
                                x,
                                # y_golden,
                                y_ndarray,
                                feat_size=feat_size,
                                bucket_widths=final_bucket_config,
                                bucketing_format=hyb_format,
                                coarsening_factor=2,
                                num_col_parts=num_parts,
                                use_implicit_unroll=args.implicit_unroll)
        exe_time = float(F"{exe_time:.6}")
    except Exception as e:
        exe_time = math.inf
        print(e, file=sys.stderr)
        # search_overhead = None
        # cost_model_config = None
        # bucket_config = None
    
    print(F"bubble_overhead: {bubble_overhead}")
    print(F"memory_overhead: {memory_overhead}")
    # print(F"rows_in_bucket: {bucket_stats.total_num_rows_in_bucket}")
    # print(F"nnz_in_bucket: {bucket_stats.total_nnz_in_bucket}")
    # print(F"elements_in_bucket: {bucket_stats.total_elements_in_bucket}")
    print(F"bucket_stats: {bucket_stats}")
    features["num_parts"] = num_parts
    features["final_bucket_config"] = str(final_bucket_config)
    features["bubble_overhead"] = bubble_overhead
    features["memory_overhead"] = memory_overhead
    features["rows_in_bucket"] = bucket_stats["total_num_rows_in_bucket"]
    features["elements_in_bucket"] = bucket_stats["total_elements_in_bucket"]
    features["density"] = bucket_stats["total_nnz_in_bucket"] / bucket_stats["total_elements_in_bucket"]
    features["exe_time_hyb(ms)"] = [exe_time]

    # #####

    # names_list, feat_sizes_list, num_partitions_list, max_bucket_width_list, exe_time_list = get_names_and_num_partitions(csv_filename)

    # # GPU device
    # print(F"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', default=0)}")

    
    # # cached_config = {}

    # for name, feat_size, num_parts, max_bucket_width, exe_time_autotuned in zip(names_list, feat_sizes_list, num_partitions_list, max_bucket_width_list, exe_time_list):
    #     print("")
    #     print(F"Going to matrix {name} ...")
    #     filename = F"/raid/peng599/scratch/spmm/data/suitesparse/{name}/{name}.mtx"
    #     g = MTX(filename)

    #     # for feat_size in [32]:
    #     # for feat_size in [32, 64, 128, 256, 512]:
    #     features = g.matrix_features()
    #     features["K"] = feat_size

    #     # # If done before, skipped
    #     name = features["name"]
    #     if check_if_done_before(name, feat_size):
    #         continue

    #     cost_model_configs_list = []
    #     bucket_configs_list = []
    #     search_overhead_list = []
    #     exe_time_list = []
    #     for mem_w, bub_w in COST_MODEL_WEIGHTS:
    #         try:
    #             cost_model_config = CostModelSettings(mem_w=mem_w, bub_w=bub_w)
    #             # if name not in cached_config:
    #             start_time = time.perf_counter()
    #             # bucket_config = search_bucket_sizes(g, num_parts)
    #             bucket_config = search_bucket_config(g, num_parts, cost_model_config)
    #             end_time = time.perf_counter()
    #             search_overhead = end_time - start_time
    #             # cached_config[name] = (bucket_config, search_overhead)
    #             # else:
    #             #     bucket_config, search_overhead = cached_config[name]
    #             # test
    #             # bucket_config = [[1, 2, 4, 8, 16, 32, 64]]
    #             # bucket_config = [[8, 128, 1024]]
    #             # end test
    #             # x = th.rand((g.num_dst_nodes(), feat_size))
    #             x = th.ones((g.num_dst_nodes(), feat_size))
    #             # y_golden = dgl.ops.copy_u_sum(g, x)
    #             y_ndarray = g.dot(x.numpy())

    #             hyb_format = build_hyb_format(g,
    #                                         bucket_config)
                
    #             print(F"#### data: {name} feat_size: {feat_size} num_partitions: {num_parts} cost_model_config: {cost_model_config} bucket_config: {bucket_config}", flush=True)

    #             exe_time = bench_hyb_with_config(g,
    #                                     x,
    #                                     # y_golden,
    #                                     y_ndarray,
    #                                     feat_size=feat_size,
    #                                     bucket_widths=bucket_config,
    #                                     bucketing_format=hyb_format,
    #                                     coarsening_factor=2,
    #                                     num_col_parts=num_parts,
    #                                     use_implicit_unroll=args.implicit_unroll)
    #             exe_time = float(F"{exe_time:.6}")
    #         except Exception as e:
    #             exe_time = math.inf
    #             print(e, file=sys.stderr)
    #             search_overhead = None
    #             cost_model_config = None
    #             bucket_config = None
    #         exe_time_list.append(exe_time)
    #         search_overhead_list.append(search_overhead)
    #         cost_model_configs_list.append(cost_model_config)
    #         bucket_configs_list.append(bucket_config)

    #     features["best_num_partitions"] = num_parts
    #     features["best_max_bucket_width"] = max_bucket_width
    #     features["exe_time_autotuned(ms)"] = exe_time_autotuned
    #     # features["bucket_widths"] = [str(bucket_config)]
    #     # features["search_time(s)"] = search_overhead
    #     # features["exe_time_searched(ms)"] = [exe_time]

    save_statistics(features)
    #     save_statistics(features,
    #                     cost_model_configs_list,
    #                     bucket_configs_list,
    #                     search_overhead_list,
    #                     exe_time_list)
