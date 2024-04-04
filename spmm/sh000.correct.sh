
for dataset in cora citeseer pubmed ppi arxiv proteins reddit; do
    echo "Going to ${dataset} ..."
    python tune_spmm_hyb.test.py -d "${dataset}"
done