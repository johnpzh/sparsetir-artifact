from matrix_market import MTX
import numpy as np
import bisect

__all__ = ["build_hyb_format"]


def get_max_bucket_width(buckets: dict):
    return max(buckets.keys)


# def modify_max_bucket_width(buckets: dict,
#                            num_parts: int):
#     curr_max_width = get_max_bucket_width(buckets)
#     # get_bubble_overhead(buckets)
#     while curr_max_width < 1:
#         # Halve
#         curr_max_width /= 2


def build_list_of_list(bucket_widths):
    # shape is [num_buckets, num_partitions]
    num_parts = len(bucket_widths)
    res = []
    for part_ind in range(num_parts):
        num_buckets = len(bucket_widths[part_ind])
        inner = [ [] for _ in range(num_buckets) ]
        res.append(inner)
    
    return res


def build_hyb_format(g: MTX,
                    #  num_parts,
                     bucket_widths):
    num_parts = len(bucket_widths)
    num_rows = g.num_src_nodes()
    num_cols = g.num_dst_nodes()
    partition_size = (num_cols + num_parts - 1) // num_parts

    # Count the degree of each row in each partition
    # degree_counter shape is partition_size * num_rows
    degree_counter = [ [0] * num_rows for i in range(num_parts) ]
    for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
        part_ind = col_ind // partition_size
        degree_counter[part_ind][row_ind] += 1
    
    # num_bucket = len(bucket_widths)
    # 3-dimentional list
    # It is a list of list with shape [num_parts, num_bucket], where the innermost element is a list
    # The inner most element needs to be a numpy ndarray in the end.
    # row_indices = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    # col_indices = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    # mask = [ [ [] for j in range(num_buckets)] for i in range(num_parts) ]
    row_indices = build_list_of_list(bucket_widths)
    col_indices = build_list_of_list(bucket_widths)
    mask = build_list_of_list(bucket_widths)
    
    for row_ind, col_ind in zip(g.coo_mtx.row, g.coo_mtx.col):
        part_ind = col_ind // partition_size
        b_widths = bucket_widths[part_ind]
        num_buckets = len(b_widths)
        degree = degree_counter[part_ind][row_ind]
        bucket_ind = bisect.bisect_left(b_widths, degree)
        if bucket_ind == num_buckets:
            # The very long row goes to the largest bucket
            bucket_ind -= 1
        bucket_width = b_widths[bucket_ind]
        need_new_row = False
        # # test
        # print(F"len(col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]): {len(col_indices[part_ind][bucket_ind])} (bucket_width - 1): {(bucket_width - 1)}")
        # # end test
        remainder = len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1)
        # remainder = len(col_indices[part_ind][bucket_ind]) % bucket_width
        if remainder == 0:
            # Current row is full, so need a new row
            need_new_row = True
        else:
            assert row_indices[part_ind][bucket_ind], F"row_indices[{part_ind}][{bucket_ind}] is empty"
            if row_ind != row_indices[part_ind][bucket_ind][-1]:
                # row_ind is changed, so need a new row
                need_new_row = True
                # Padding current row if not full
                for _ in range(remainder, bucket_width):
                    col_indices[part_ind][bucket_ind].append(0)
                    mask[part_ind][bucket_ind].append(0)
        
        if need_new_row:
            # Current row is full, or the row_ind changed
            assert len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1) == 0, F"invalid padding for col_indices[{part_ind}][{bucket_ind}] and bucket_width={bucket_width}"
            # assert len(col_indices[part_ind][bucket_ind]) % bucket_width == 0, F"invalid padding for col_indices[{part_ind}][{bucket_ind}] and bucket_width={bucket_width}"
            row_indices[part_ind][bucket_ind].append(row_ind)
        
        col_indices[part_ind][bucket_ind].append(col_ind)
        mask[part_ind][bucket_ind].append(1)

    # # test
    # for part_ind in range(num_parts):
    #     b_widths = bucket_widths[part_ind]
    #     num_buckets = len(b_widths)
    #     print(F"\npart_ind: {part_ind} num_buckets: {num_buckets}")
    #     for bucket_ind in range(num_buckets):
    #         bucket_width = bucket_widths[part_ind][bucket_ind]
    #         print(F"\nbucket_ind: {bucket_ind} bucket_width: {bucket_width}")
    #         print(F"row_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]: {row_indices[part_ind][bucket_ind]}")
    #         print(F"col_indices[part_ind={part_ind}][bucket_ind={bucket_ind}]: {col_indices[part_ind][bucket_ind]}")
    #         print(F"mask[part_ind={part_ind}][bucket_ind={bucket_ind}]: {mask[part_ind][bucket_ind]}")
    # # end test

    # Padding the last rows and convert to numpy ndarray required by SparseTIR
    row_indices_nd = []
    col_indices_nd = []
    mask_nd = []
    for part_ind in range(num_parts):
        row_indices_part_local = []
        col_indices_part_local = []
        mask_part_local = []
        b_widths = bucket_widths[part_ind]
        num_buckets = len(b_widths)
        for bucket_ind in range(num_buckets):
            bucket_width = b_widths[bucket_ind]
            remainder = len(col_indices[part_ind][bucket_ind]) & (bucket_width - 1)
            # remainder = len(col_indices[part_ind][bucket_ind]) % bucket_width
            # Padding the last row
            if remainder:
                for _ in range(remainder, bucket_width):
                    col_indices[part_ind][bucket_ind].append(0)
                    mask[part_ind][bucket_ind].append(0)

            num_nz_rows = len(row_indices[part_ind][bucket_ind])
            # # test
            # print(F"len(col_indices[{part_ind}][{bucket_ind}]): {len(col_indices[part_ind][bucket_ind])}")
            # print(F"num_nz_rows: {num_nz_rows} bucket_width: {bucket_width} num_nz_rows * bucket_width: {num_nz_rows * bucket_width}")
            # # end test
            assert len(col_indices[part_ind][bucket_ind]) == num_nz_rows * bucket_width, F"invalid padding for len(col_indices[{part_ind}][{bucket_ind}]) is not equal to num_nz_rows * bucket_width ({num_nz_rows} * {bucket_width})"
            assert len(mask[part_ind][bucket_ind]) == num_nz_rows * bucket_width, F"invalid padding for len(mask[{part_ind}][{bucket_ind}]) is not equal to num_nz_rows * bucket_width ({num_nz_rows} * {bucket_width})"
            
            # Convert to numpy ndarray
            if num_nz_rows:
                row_indices_part_local.append(np.array(row_indices[part_ind][bucket_ind]))
                col_indices_part_local.append(np.array(col_indices[part_ind][bucket_ind]))
                mask_part_local.append(np.array(mask[part_ind][bucket_ind]))
            else:
                row_indices_part_local.append(np.empty(num_nz_rows, dtype=int))
                col_indices_part_local.append(np.empty(num_nz_rows * bucket_width, dtype=int))
                mask_part_local.append(np.empty(num_nz_rows * bucket_width, dtype=int))
        row_indices_nd.append(row_indices_part_local)
        col_indices_nd.append(col_indices_part_local)
        mask_nd.append(mask_part_local)

    # # test
    # for part_ind in range(num_parts):
    #     b_widths = bucket_widths[part_ind]
    #     num_buckets = len(b_widths)
    #     print(F"\npart_ind: {part_ind} num_buckets: {num_buckets}")
    #     for bucket_ind in range(num_buckets):
    #         bucket_width = bucket_widths[part_ind][bucket_ind]
    #         print(F"\nbucket_ind: {bucket_ind} bucket_width: {bucket_width}")
    #         print(F"row_indices_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {row_indices_nd[part_ind][bucket_ind]}")
    #         print(F"col_indices_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {col_indices_nd[part_ind][bucket_ind]}")
    #         print(F"mask_nd[part_ind={part_ind}][bucket_ind={bucket_ind}]: {mask_nd[part_ind][bucket_ind]}")
    # # end test
            
    return (row_indices_nd, col_indices_nd, mask_nd)
        