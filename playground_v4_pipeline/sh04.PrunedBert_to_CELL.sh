python py03.PrunedBert_structured_to_CELL.py \
-s models/output_2279_total.for_selection.train_set.RandomForest.joblib \
-p models/output_2279_total.for_partitions.train_set.RandomForest.joblib \
2>&1 | tee output.py03.PrunedBert_structured_to_CELL.py.log

python py04.PrunedBert_unstructured_to_CELL.py \
-s models/output_2279_total.for_selection.train_set.RandomForest.joblib \
-p models/output_2279_total.for_partitions.train_set.RandomForest.joblib \
2>&1 | tee output.py04.PrunedBert_unstructured_to_CELL.py.log