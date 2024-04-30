#include <vector>

using std::vector;
// Cost(m, w, T)
class Bucket{
    int w;
    vector<int> row_ids;
    vector< vector<int> > col_ids;
};

/// @brief search for a better bucket to hold input rows
/// @param m: number of rows 
/// @param w: bucket width
/// @param T: number of non-zeros
/// @param bucket: bucket data structure
/// @return a new bucket
bucket search(m, w, T, bucket) {
    potential_w = [1, 2, 4, ..., w];
    cur_cost = cost(m, w, T);
    for (new_w in potential_w) {
        new_m = get_num_rows(new_w, bucket);
        new_cost = cost(new_m, new_w, T);
        if (new_cost < cur_cost) {
            new_bucket = build_bucket(new_w, bucket);
            has_new = true;
            break;
        }
    }

    if (has_new) {
        return new_bucket;
    } else {
        return bucket;
    }
}

buckets decide_bucket(matrix) {
    // put all rows into its corresponding initial bucket whose size is a power of 2;
    init_buckets = get_init_bucket(matrix);

    // Search for better buckets
    for (bucket in init_buckets) {
        w = bucket.width;
        m = bucket.num_rows;
        nnz = bucket.nnz;
        new_bucket = search(m, w, nnz, bucket);
        buckets.add(new_bucket);
    }

    return buckets;
}