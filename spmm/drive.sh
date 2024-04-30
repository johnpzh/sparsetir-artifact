
# bash sh002.search_on_gpu2_part1.sh & \
# bash sh002.search_on_gpu2_part2.sh & \
# bash sh002.search_on_gpu2_part3.sh & \
# bash sh002.search_on_gpu2_part4.sh & \
# bash sh002.search_on_gpu2_part5.sh & \
# bash sh002.search_on_gpu2_part6.sh

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="7"

python bench_suitesparse_spmm_hyb.py -d ../data/suitesparse/bcsstk06/bcsstk06.mtx