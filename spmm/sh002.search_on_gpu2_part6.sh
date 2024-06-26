
ulimit -s unlimited

export FLUSH_L2=ON

data_dir="../data/suitesparse"
start_time=$(date +%s)

gpu_index=7

app="bench_suitesparse_spmm_hyb.searchv2.gpu_index.py"

#### Import MATRICES
# source "dataset_names_test.sh"
# source "dataset_names_242_part6.sh"
source "dataset_names_242_part6.v2.sh"

# for name in cora citeseer; do
# for name in cora citeseer pubmed ppi; do
# for name in cora citeseer pubmed ppi arxiv proteins reddit; do
for mtx in "${MATRICES[@]}"; do
    name=$(basename "${mtx}" .tar.gz)
    echo ""
    echo "Going to ${name} ..."
    python "${app}" -d "${data_dir}/${name}/${name}.mtx" -g "${gpu_index}" 2>&1 | tee "output/output_tune_${name}_hyb_verbose.log"
    python extract_data.search.py -d "${name}"
done

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"