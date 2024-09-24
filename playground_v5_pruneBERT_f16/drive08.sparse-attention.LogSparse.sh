# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES="0"

# export FLUSH_L2=ON

python proc08.sparse-attention.LogSparse.py \
-d LogSparse \
-s models/output_2279_total.for_selection.train_set.RandomForest.joblib \
-p models/output_2279_total.for_partitions.train_set.RandomForest.joblib 