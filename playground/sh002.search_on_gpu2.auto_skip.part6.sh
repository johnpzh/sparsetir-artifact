
ulimit -s unlimited

export FLUSH_L2=ON

data_dir="../data/suitesparse"
start_time=$(date +%s)

# gpu_index=6
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="6"

app="bench_suitesparse_spmm_hyb.search.v3.auto_skip.py"

#### Import MATRICES
source "dataset_names_2904_part6.txt"
# source "dataset_names_242.sh"
gpu_log="output/output_gpu${CUDA_VISIBLE_DEVICES}_log.log"
:> ${gpu_log}
# for name in cora citeseer; do
# for name in cora citeseer pubmed ppi; do
# for name in cora citeseer pubmed ppi arxiv proteins reddit; do
for mtx in "${MATRICES[@]}"; do
    name=$(basename "${mtx}" .tar.gz)
    echo "" | tee -a "${gpu_log}"
    echo "Going to ${name} ..." | tee -a "${gpu_log}"
    python "${app}" -d "${data_dir}/${name}/${name}.mtx" 2>&1 | tee "output/output_tune_${name}_hyb_verbose.log"
    python extract_data.search.py -d "${name}" | tee -a "${gpu_log}"
done

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "" | tee -a "${gpu_log}"
echo "####" | tee -a "${gpu_log}"
echo "#### Execution_time(s): ${runtime}" | tee -a "${gpu_log}"