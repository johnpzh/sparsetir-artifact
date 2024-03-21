# # Run SpMM experiments
# docker run -it --gpus all -v $(pwd)/spmm/:/root/spmm sparsetir /bin/bash -c 'cd spmm && bash run.sh'
# # Run SDDMM experiments
# docker run -it --gpus all -v $(pwd)/sddmm/:/root/sddmm sparsetir /bin/bash -c 'cd sddmm && bash run.sh'
# # Run GraphSAGE training experiments
# docker run -it --gpus all -v $(pwd)/e2e/:/root/e2e sparsetir /bin/bash -c 'cd e2e && bash run.sh'
# # Run RGCN inference experiments
# docker run -it --gpus all -v $(pwd)/rgcn/:/root/rgcn sparsetir /bin/bash -c 'cd rgcn && bash run.sh'
# # Run Sparse Attention experiments
# docker run -it --gpus all -v $(pwd)/sparse-attention/:/root/sparse-attention sparsetir /bin/bash -c 'cd sparse-attention && bash run.sh'
# # Run PrunedBERT experiments
# docker run -it --gpus all -v $(pwd)/pruned-bert/:/root/pruned-bert sparsetir /bin/bash -c 'cd pruned-bert && bash run.sh'
# # Run Sparse Convolution experiments
# docker run -it --gpus all -v $(pwd)/sparse-conv/:/root/sparse-conv sparsetir /bin/bash -c 'cd sparse-conv && bash run.sh'

# docker run -it --gpus all \
# -v $(pwd)/spmm/:/root/spmm \
# -v $(pwd)/sddmm/:/root/sddmm \
# -v $(pwd)/e2e/:/root/e2e \
# -v $(pwd)/rgcn/:/root/rgcn \
# -v $(pwd)/sparse-attention/:/root/sparse-attention \
# -v $(pwd)/pruned-bert/:/root/pruned-bert \
# -v $(pwd)/sparse-conv/:/root/sparse-conv \
# -v $(pwd)/sh-run-all.sh:/root/sh-run-all.sh
# sparsetir /bin/bash

# docker run -it --gpus all -v "$(pwd)":/root/sparsetir sparsetir /bin/bash

# Run SpMM experiments
cd spmm && bash run.sh && cd ..
# # Run SDDMM experiments
cd sddmm && bash run.sh && cd ..
# # Run GraphSAGE training experiments
cd e2e && bash run.sh && cd ..
# # Run RGCN inference experiments
cd rgcn && bash run.sh && cd ..
# # Run Sparse Attention experiments
cd sparse-attention && bash run.sh && cd ..
# # Run PrunedBERT experiments
cd pruned-bert && bash run.sh && cd ..
# # Run Sparse Convolution experiments
cd sparse-conv && bash run.sh && cd ..
