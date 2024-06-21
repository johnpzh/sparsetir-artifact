
start_time=$(date +%s)

bash drive05.bench_hyb.search.v9.cost_model.profile_w1.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w2.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w4.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w8.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w16.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w32.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w64.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w128.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w256.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w512.sh && \
bash drive05.bench_hyb.search.v9.cost_model.profile_w1024.sh

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo ""
echo "Execution_time(s): ${runtime}"
