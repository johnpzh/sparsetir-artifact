
ulimit -s unlimited

export FLUSH_L2=ON


start_time=$(date +%s)

# for name in cora citeseer; do
for name in arxiv proteins reddit; do
# for name in cora citeseer pubmed ppi arxiv proteins reddit; do
    echo ""
    echo "Going to ${name} ..."
    python bench_suitesparse_spmm_hyb.search.py -d "../data/${name}/${name}.mtx" 2>&1 | tee "output/output_tune_${name}_verbose.log"
    python extract_data.search.py -d "${name}"
done

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"