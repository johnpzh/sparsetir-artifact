
ulimit -s unlimited

export FLUSH_L2=ON


start_time=$(date +%s)

# gpu_index=1

#### Import MATRICES
# source "dataset_names_test.sh"
source "dataset_names_242.sh"

# for name in cora citeseer; do
# for name in cora citeseer pubmed ppi; do
# for name in cora citeseer pubmed ppi arxiv proteins reddit; do
for mtx in "${MATRICES[@]}"; do
    name=$(basename "${mtx}" .tar.gz)
    echo ""
    echo "Collecting ${name} hyb and no-hyb together ..."
    python extract_data.hyb_and_naive.py -d "${name}"
done

output_dir="output"
is_first=1
total_csv="${output_dir}/output_0_total.csv"
# for name in cora citeseer pubmed ppi; do
for mtx in "${MATRICES[@]}"; do
    name=$(basename "${mtx}" .tar.gz)
    input_csv="${output_dir}/output_tune_${name}_hyb-naive_collect.csv"
    if [ ${is_first} -eq 1 ]; then
        is_first=0
        head -n 1 "${input_csv}" > "${total_csv}"
    fi
    tail -n 5 "${input_csv}" >> "${total_csv}"
done

echo ""
echo "Saved to ${total_csv} ."

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"