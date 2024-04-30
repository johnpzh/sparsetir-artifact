output_dir="output"



for GPU in 1 2 3 4 5 6; do
    echo ""
    echo "GPU:${GPU}"
    source "dataset_names_1834_part${GPU}.txt"
    ready=0
    count=0
    is_first=1
    for name in "${MATRICES[@]}"; do
        ((++count))
        if [[ (! -f "${output_dir}/output_tune_${name}_hyb_collect.csv") && (${is_first} -eq 1) ]]; then
            # is_first=0
            echo "Matrix ${name} is not ready."
        elif [[ -f "${output_dir}/output_tune_${name}_hyb_collect.csv" ]]; then
            ((++ready))
        fi
    done
    echo "GPU ${GPU} has ${ready}/${count} ready."
done

