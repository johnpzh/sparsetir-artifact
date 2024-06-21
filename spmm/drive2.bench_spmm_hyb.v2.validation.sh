# Environments
ulimit -s unlimited
export FLUSH_L2=ON
# GPU
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="6"

app="bench_spmm_hyb.v2.validation.py"
# csv_feature="suitesparse_3_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_333_csr_052324.drop_duplicate_names.csv"
# csv_feature="suitesparse_1683_csr_052324.drop_duplicate_names.csv"
# data_dir="../data/suitesparse"
# name=ins2
# name=G36
# name="dc2"
name="GD96_b"
# mtx_file="${data_dir}/${name}/${name}.mtx"
# mtx_file="../data/correct/correct.square.mtx"
# mtx_file="../data/correct/correct.dumb.mtx"
# mtx_file="../data/correct/correct.synthesize.mtx"
# new_max_width=512
# new_max_width=1
num_parts=2
feat_size=32
output_dir="output.validation"
verbose_file="${output_dir}/output_0_original_hyb_verbose.log"
:> "${verbose_file}"

start_time=$(date +%s)

# Kernel
echo "" | tee -a "${verbose_file}"
echo "Run kernel ..." | tee -a "${verbose_file}"

# python "${app}" -f "${csv_feature}" 2>&1 | tee -a "${verbose_file}"
python "${app}" -d "${name}" -i 2>&1 | tee -a "${verbose_file}"

# echo "" | tee -a "${verbose_file}"
# echo "Combine results ..." | tee -a "${verbose_file}"

# python py00.extract_hyb_searched_results.py -f "${csv_feature}" 2>&1 | tee -a "${verbose_file}"

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "" | tee -a "${verbose_file}"
echo "Execution_time(s): ${runtime}" | tee -a "${verbose_file}"
