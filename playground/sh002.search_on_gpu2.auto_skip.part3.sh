
ulimit -s unlimited

export FLUSH_L2=ON

data_dir="../data/suitesparse"
start_time=$(date +%s)

# gpu_index=3
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

app="bench_suitesparse_spmm_hyb.search.v3.auto_skip.py"

#### Import MATRICES
source "dataset_names_2904_part3.txt"
# source "dataset_names_242.sh"

# for name in cora citeseer; do
# for name in cora citeseer pubmed ppi; do
# for name in cora citeseer pubmed ppi arxiv proteins reddit; do
for mtx in "${MATRICES[@]}"; do
    name=$(basename "${mtx}" .tar.gz)
    echo ""
    echo "Going to ${name} ..."
    python "${app}" -d "${data_dir}/${name}/${name}.mtx" 2>&1 | tee "output/output_tune_${name}_hyb_verbose.log"
    python extract_data.search.py -d "${name}"
done

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"