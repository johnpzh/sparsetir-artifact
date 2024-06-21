# Environments
ulimit -s unlimited
export FLUSH_L2=ON
# GPU
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="2"

app="bench_suitesparse_spmm_hyb.search.v11.memory_cost_model.from_csv.py"
# csv_feature="suitesparse_3_single_sp-ge-1.1_050424.csv"
# csv_feature="suitesparse_122_csr_052324.drop_duplicate_names.csv"
# csv_feature="suitesparse_996_csr_052324.drop_duplicate_names.csv"
csv_feature="suitesparse_1683_csr_052324.num_parts_gt_one.csv"
# csv_feature="output_22_selected_total.csv"

output_dir="output.cost_model"
verbose_file="${output_dir}/output_0_hyb_searched_verbose.log"
:> "${verbose_file}"

start_time=$(date +%s)

# Kernel
echo "" | tee -a "${verbose_file}"
echo "Run kernel ..." | tee -a "${verbose_file}"

python "${app}" -f "${csv_feature}" 2>&1 | tee -a "${verbose_file}"

echo "" | tee -a "${verbose_file}"
echo "Combine results ..." | tee -a "${verbose_file}"

python py00.extract_hyb_searched_results.py -f "${csv_feature}" 2>&1 | tee -a "${verbose_file}"

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "" | tee -a "${verbose_file}"
echo "Execution_time(s): ${runtime}" | tee -a "${verbose_file}"
