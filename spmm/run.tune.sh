#!/bin/bash

start_time=$(date +%s)

export FLUSH_L2=ON
ulimit -s unlimited

# Benchmark
# echo "Running SpMM benchmark..."
# if [ ! -d data/ ]
# then
#   python3 dump_npz.py > dump_npz.log 2> dump_npz.err
# fi

for dataset in proteins reddit
# for dataset in cora citeseer pubmed ppi arxiv proteins reddit
do
  echo "Running tuning SpMM w/ hybrid format on ${dataset}"
  python3 tune_spmm_hyb.py -d ${dataset} -i > logs_tune_${dataset}_hyb.log 2> errors_tune_${dataset}_hyb.log
done

# # Extract data
# echo "Extracting data from log files..."
# python3 extract_data.TEST.py

# echo "Done. See spmm.TEST.csv for results."

# # Plot figures
# echo "Plotting figures..."
# python3 plot.py

# echo "Evaluation finished, see spmm.pdf for results."

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "####"
echo "#### Execution_time(s): ${runtime}"
