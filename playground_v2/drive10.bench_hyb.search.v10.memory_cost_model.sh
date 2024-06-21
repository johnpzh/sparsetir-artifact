# Environments
ulimit -s unlimited
export FLUSH_L2=ON
# GPU
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="6"

app="bench_suitesparse_spmm_hyb.search.v10.memory_cost_model.py"
# csv_feature="suitesparse_108_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_1083_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_27_crashed_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_incorrect_3_matrices_050224.csv"
# csv_feature="suitesparse_3_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_24_microbench_single_sp-ge-1.1_050424.csv"
data_dir="../data/suitesparse"
######################################
# tols2000: [1, 2, 128] vs. 2, too large width
# bayer08: [1, 2, 4, 8, 16, 32, 64] vs. 8, too large width
# fem_filter: [8, 16, 32, 1024] vs. 16, too large width
# brainpc2: [2, 4] vs. 32, too small width
# ins2: [8] vs. 64, too small width
# dc2: [1, 2, 4, 8] vs. 64, too small width
# boyd2: [2, 4] vs. 64, too small width
######################################
# name="tols2000"
# name="bayer08"
# name="fem_filter"
# name="TSOPF_RS_b162_c1"
# name="boyd1"
name="brainpc2"
# name="ins2"
# name="dc2"
# name="G36"
# name="GD96_b"
mtx_file="${data_dir}/${name}/${name}.mtx"
# mtx_file="../data/correct/correct.square.mtx"
# mtx_file="../data/correct/correct.dumb.mtx"
# mtx_file="../data/correct/correct.synthesize.mtx"
# new_max_width=512
# new_max_width=1
new_max_width=1
num_parts=1
# num_parts=8
output_dir="output.cost_model_profile"
verbose_file="${output_dir}/output_0_${name}_hyb_profile_verbose.log"
:> "${verbose_file}"

start_time=$(date +%s)

# Kernel
echo "" | tee -a "${verbose_file}"
echo "Run kernel ..." | tee -a "${verbose_file}"

python "${app}" -f "${mtx_file}" -w "${new_max_width}" -p "${num_parts}" 2>&1 | tee -a "${verbose_file}"

# echo "" | tee -a "${verbose_file}"
# echo "Combine results ..." | tee -a "${verbose_file}"

# python py01.combine_microbench_results.py -f "${csv_feature}" 2>&1 | tee -a "${verbose_file}"

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "" | tee -a "${verbose_file}"
echo "Execution_time(s): ${runtime}" | tee -a "${verbose_file}"
