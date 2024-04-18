
ulimit -s unlimited

export FLUSH_L2=ON


start_time=$(date +%s)

# gpu_index=1

#### Import MATRICES
# source "dataset_names_test.sh"
source "dataset_names_1834.txt"
output_dir="output"

# for name in cora citeseer; do
# for name in cora citeseer pubmed ppi; do
# for name in cora citeseer pubmed ppi arxiv proteins reddit; do
ready=0
total=0
for mtx in "${MATRICES[@]}"; do
    ((total++))
    name=$(basename "${mtx}" .tar.gz)
    hyb_file="${output_dir}/output_tune_${name}_hyb_collect.csv"
    naive_file="${output_dir}/output_tune_${name}_naive_collect.csv"
    if [ ! -f "${hyb_file}" ]; then
        # Some hyb version performance is not ready yet.
        echo "${name} hyb not ready, yet. Passed"
        continue
    elif [ ! -f "${naive_file}" ]; then
        # Some naive version performance is not ready yet.
        echo "${name} naive not ready, yet. Passed"
        continue
    fi
    ((ready++))
    echo ""
    echo "Collecting ${name} hyb and no-hyb together ..."
    python extract_data.hyb_and_naive.py -d "${name}"
done

is_first=1
total_csv="${output_dir}/output_0_total.csv"
# for name in cora citeseer pubmed ppi; do
for mtx in "${MATRICES[@]}"; do
    name=$(basename "${mtx}" .tar.gz)

    input_csv="${output_dir}/output_tune_${name}_hyb-naive_collect.csv"
    if [ ! -f "${input_csv}" ]; then
        # Some performance is not ready yet.
        continue
    fi
    if [ ${is_first} -eq 1 ]; then
        is_first=0
        head -n 1 "${input_csv}" > "${total_csv}"
    fi
    tail -n 5 "${input_csv}" >> "${total_csv}"
done

echo ""
echo "Saved to ${total_csv} ."
echo "ready/total: ${ready}/${total}"

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"