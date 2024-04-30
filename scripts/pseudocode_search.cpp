
/// @brief search the best configuration. A config has 
/// 1) a number of paritions, 
/// 2) a maximum bucket width, and 
/// 3) corresponding execution time.
/// @param  config_s: the starting configuration
/// @return the best configuration that has the shortest execution time
configuration search(config_s) {
    curr = config_s;
    curr.exe_time = spmm(curr);
    is_visited.add(curr);
    stack.push(curr);
    best_config = curr;

    while (!stack.empty()) {
        curr = stack.pop();
        for (each next configs from curr) {
            nxt = curr.next();
            if (nxt in is_visited) {
                continue;
            }
            is_visited.add(nxt);
            nxt.exe_time = spmm(nxt);
            if (nxt.exe_time > curr.exe_time) {
                // Pruning
                continue;
            }
            stack.push(nxt);
            if (nxt.exe_time < best_config.exe_time) {
                // Record the best config so fa
                best_config = nxt;
            }
        }
    }

    return best_config;
}

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


    
