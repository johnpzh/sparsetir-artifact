feature_file_name="output_0_total"
raw_data_file="../spmm/output/${feature_file_name}.csv"
# feature_file_name="dummy"
# raw_data_file="data/${feature_file_name}.csv"
train_set_file="data/${feature_file_name}.train_set.csv"
test_set_file="data/${feature_file_name}.test_set.csv"


# # Split the dataset
# sed -i "s/_densitry_/_density_/g" "${raw_data_file}"

# echo ""
# echo "Splitting the dataset ..."
# python proc_split_train_test.py -f "${raw_data_file}" -r 0.9

###############################
# RandomForestClassifier.density
###############################

# Train the model
echo ""
echo "Training the model ..."
python proc_binary_desicion.v1.RandomForestClassifier.density.fit.py -f "${train_set_file}"

# Test the model
echo ""
echo "Testing the model ..."
model_file="output/${feature_file_name}.train_set.RandomForestClassifier.density.joblib"
python proc_binary_desicion.v1.RandomForestClassifier.density.predict.py -f "${test_set_file}" -m "${model_file}"

###############################
# RandomForestClassifier.abs_value
###############################

# Train the model
echo ""
echo "Training the model ..."
python proc_binary_desicion.v2.RandomForestClassifier.abs_value.fit.py -f "${train_set_file}"

# Test the model
echo ""
echo "Testing the model ..."
model_file="output/${feature_file_name}.train_set.RandomForestClassifier.abs_value.joblib"
python proc_binary_desicion.v2.RandomForestClassifier.abs_value.predict.py -f "${test_set_file}" -m "${model_file}"

# ###############################
# # LogisticRegression.density
# ###############################

# # Train the model
# echo ""
# echo "Training the model ..."
# python proc_binary_desicion.v3.LogisticRegression.density.fit.py -f "${train_set_file}"

# # Test the model
# echo ""
# echo "Testing the model ..."
# model_file="output/${feature_file_name}.train_set.LogisticRegression.density.joblib"
# python proc_binary_desicion.v3.LogisticRegression.density.predict.py -f "${test_set_file}" -m "${model_file}"

# ###############################
# # LogisticRegression.abs_value
# ###############################

# # Train the model
# echo ""
# echo "Training the model ..."
# python proc_binary_desicion.v4.LogisticRegression.abs_value.fit.py -f "${train_set_file}"

# # Test the model
# echo ""
# echo "Testing the model ..."
# model_file="output/${feature_file_name}.train_set.LogisticRegression.abs_value.joblib"
# python proc_binary_desicion.v4.LogisticRegression.abs_value.predict.py -f "${test_set_file}" -m "${model_file}"