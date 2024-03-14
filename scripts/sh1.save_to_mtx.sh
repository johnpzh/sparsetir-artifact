# datasets=(\
# "cora" \
# )
datasets=(\
"arxiv" \
"proteins" \
"pubmed" \
"citeseer" \
"cora" \
"ppi" \
"reddit" \
"products" \
)

start_time=$(date +%s)

data_dir="/raid/peng599/spmm/datasets"

for dataset in "${datasets[@]}"; do
    echo ""
    echo "#### Going to dataset ${dataset}"
    dir="${data_dir}/${dataset}"
    if [ ! -d "${dir}" ]; then
        mkdir -p "${dir}"
    fi
    python3 save_to_mtx.py ${dataset} "${dir}/${dataset}.mtx"
done

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"