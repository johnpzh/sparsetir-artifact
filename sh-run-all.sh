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
