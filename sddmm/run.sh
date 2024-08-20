#!/bin/bash

# Benchmark
echo "Running SDDMM benchmark..."
if [ ! -d data/ ]
then
  python3 dump_npz.py > dump_npz.log 2> dump_npz.err
fi

for dataset in cora #citeseer pubmed ppi arxiv proteins reddit
do
  # sparsetir & dgl
  echo "Running SparseTIR SDDMM on ${dataset}"
  python3 bench_sddmm_ell.py -d ${dataset} #> sparsetir_${dataset}.log 2> sparsetir_${dataset}.err
  # echo "Running DGL SDDMM on ${dataset}"
  # python3 bench_dgl.py -d ${dataset} > dgl_${dataset}.log 2> dgl_${dataset}.err
  # for feat_size in 32 64 128 256 512
  # do
  #   # dgsparse
  #   echo "Running dgsparse SDDMM on ${dataset}, feat_size = ${feat_size}"
  #   dgsparse-sddmm data/${dataset}-sddmm.npz ${feat_size} > dgsparse_${dataset}_${feat_size}.log 2> dgsparse_${dataset}_${feat_size}.err
  #   # sputnik
  #   echo "Running sputnik SDDMM on ${dataset}, feat_size = ${feat_size}"
  #   sputnik_sddmm_benchmark data/${dataset}-sddmm.npz ${feat_size} > sputnik_${dataset}_${feat_size}.log 2> sputnik_${dataset}_${feat_size}.err
  #   # taco
  #   echo "Running taco SDDMM on ${dataset}, feat_size = ${feat_size}"
  #   taco-sddmm data/${dataset}-sddmm.npz ${feat_size} > taco_${dataset}_${feat_size}.log 2> taco_${dataset}_${feat_size}.err
  # done
done

# Extract data
echo "Extracting data from log files..."
python3 extract_data.py

# Plot figures
echo "Plotting figures..."
python3 plot.py

echo "Evaluation finished, see sddmm.pdf for results."
