export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="6"

export FLUSH_L2=ON

python proc15.spmm.suitesparse.max_bucket.part1.py \
-c data/output_2279_total.no_nan.csv \
-s models/output_2279_total.for_selection.train_set.RandomForest.joblib \
-p models/output_2279_total.for_partitions.train_set.RandomForest.joblib