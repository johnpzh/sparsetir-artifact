

# void search(Matrix matrix) {
#     int x = 0;
#     int y = 0;
#     int num_cols;
#     int num_rows;
#     Point curr(0, 0);
#     bool is_visited = {false};
#     vector<point> queue;
#     double min_val = curr.val;
#     Point min_coordinate = curr.corrdinate;
#     queue.push_back(curr);

#     while (!queue.empty()) {
#         curr = queue.back();
#         queue.pop_back();
#         for (n in  curr neighbors) {
#             if (n is not inside the matrix) {
#                 continue;
#             }
#             if (n is visited) {
#                 continue;
#             }
#             is_visited[n] = true;
#             if (n.val > curr.val) {
#                 continue;
#             }
#             queue.push_back(n);
#             if (matrix[n].val < min_val) {
#                 min_val = matrix[n].val;
#                 min_coordinate = n;
#             }
#         }
#     }
# }

def search_exe_time():
    Config curr(1, 1)
    curr.exe_time = bench(curr)
    is_visited = [curr]
    queue = [curr]
    min_config = curr

    while queue:
        curr = queue.pop()
        for step in directions:
            next = Config(curr + step)
            if next.num_parts < 1 or next.max_bucket_width < 1:
                continue
            if next in is_visited:
                continue
            is_visited.append(next) # visited
            next.exe_time = bench(next) # measure time
            if next.exe_time > curr.exe_time:
                continue
            queue.append(next)
            if next.exe_time < min_config.exe_time:
                min_config = next


    
