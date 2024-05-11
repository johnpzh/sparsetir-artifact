# Environments
ulimit -s unlimited
# export FLUSH_L2=ON
export FLUSH_L2=OFF
# GPU
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

# name="HTC_336_4438"
# name="ins2"
# name="ASIC_100k"
# name="bloweybq"
# name="cage3"
# name="adder_dcop_09"
# name="ins2"
# name="dc2"
app="bench_suitesparse_spmm_hyb.correctness.v2.cost_model.py"
output_dir="output.cost_model"
# data_file="../data/suitesparse/${name}/${name}.mtx"
# data_file="../data/correct/correct.square.mtx"
# data_file="../data/correct/correct.cage3.mtx"
# data_file="../data/correct/correct.b1_ss.mtx"
data_file="../data/correct/correct.dumb.mtx"
num_parts=1

start_time=$(date +%s)

# Kernel
python "${app}" -d "${data_file}" 2>&1 | tee "${output_dir}/output_tune_${name}_hyb_verbose.log"

# Walltime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "Execution_time(s): ${runtime}"