export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

ulimit -s unlimited
export PATH="/usr/local/cuda/bin:${PATH}"
export TMPDIR="/raid/peng599/scratch/tmp"
export FLUSH_L2=ON

/home/peng599/local/repos/miniconda3/envs/sparsetir/bin/python proc15.profile.part.width.py -p 32 -w 32