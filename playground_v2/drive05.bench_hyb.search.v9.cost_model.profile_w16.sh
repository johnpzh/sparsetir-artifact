# Environments
ulimit -s unlimited
export FLUSH_L2=ON
export PATH="/usr/local/cuda/bin:${PATH}"
export TMPDIR="/raid/peng599/scratch/tmp"
# GPU
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

app="bench_suitesparse_spmm_hyb.search.v9.cost_model.profile.py"
# csv_feature="suitesparse_108_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_1083_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_27_crashed_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_incorrect_3_matrices_050224.csv"
# csv_feature="suitesparse_3_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_24_microbench_single_sp-ge-1.1_050424.csv"
data_dir="../data/suitesparse"
# name=ins2
# name=G36
# mtx_file="${data_dir}/${name}/${name}.mtx"
# mtx_file="../data/correct/correct.square.mtx"
# mtx_file="../data/correct/correct.dumb.mtx"
mtx_file="../data/correct/correct.synthesize.mtx"
# new_max_width=512
# new_max_width=1
new_max_width=16
output_dir="output.cost_model_profile"
verbose_file="${output_dir}/output_0_hyb_profile_verbose.w16.log"
:> "${verbose_file}"

start_time=$(date +%s)

# Kernel
echo "" | tee -a "${verbose_file}"
echo "Run kernel ..." | tee -a "${verbose_file}"

/home/peng599/local/repos/miniconda3/envs/sparsetir/bin/python "${app}" -f "${mtx_file}" -w "${new_max_width}" 2>&1 | tee -a "${verbose_file}"

# echo "" | tee -a "${verbose_file}"
# echo "Combine results ..." | tee -a "${verbose_file}"

# python py01.combine_microbench_results.py -f "${csv_feature}" 2>&1 | tee -a "${verbose_file}"

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "" | tee -a "${verbose_file}"
echo "Execution_time(s): ${runtime}" | tee -a "${verbose_file}"
