from os.path import basename, splitext
from scipy.io import mmread
import numpy as np
import math

__all__ = ["MTX", "Bucket"]

class Bucket:
    def __init__(self,
                 bucket_width: int,
                 num_rows: int,
                 nnz: int):
        self.bucket_width = bucket_width
        self.num_rows = num_rows
        self.nnz = nnz
    
    def __str__(self):
        return F"Bucket(bucket_width={self.bucket_width}, num_rows={self.num_rows}, nnz={self.nnz})"


class MTX:
    def __init__(self, 
                 filename: str) -> None:
        self.name = splitext(basename(filename))[0]
        self.coo_mtx = mmread(filename)
        # Fill data to all ones, in order to be compatible with the correctness check in SparseTIR.
        for i in range(len(self.coo_mtx.data)):
            self.coo_mtx.data[i] = 1
        pass

    def adj_tensors(self,
                    format: str):
        if format == "coo":
            return self.coo_mtx.row, self.coo_mtx.col
        elif format == "csc":
            csc_mtx = self.coo_mtx.tocsc()
            return csc_mtx.indptr, csc_mtx.indices, csc_mtx.data
        elif format == "csr":
            csr_mtx = self.coo_mtx.tocsr()
            return csr_mtx.indptr, csr_mtx.indices, csr_mtx.data
        else:
            raise TypeError(F"Format {format} is not suppoted.")

    def num_src_nodes(self):
        return self.coo_mtx.shape[0]
    
    def num_dst_nodes(self):
        return self.coo_mtx.shape[1]
    
    def num_edges(self):
        return self.coo_mtx.nnz
    
    def dot(self, B):
        """Return multiplication (A * B) using SciPy dot. 
        B is a dense matrix (numpy ndarray).

        Args:
            B (numpy ndarray): one dense matrix

        Returns:
            numpy ndarray: multiplication result C = A * B
        """
        return self.coo_mtx.tocsc().dot(B)
        # return self.coo_mtx.dot(B)
    
    def matrix_features(self):
        """Return matrix's features in a dict.

        Returns:
            dict: features including
                * name
                * num_rows
                * num_cols
                * nnz
                * avg_nnz_per_row
                * min_nnz_per_row
                * max_nnz_per_row
                * std_dev_nnz_per_row
                * avg_nnz_density_per_row
                * min_nnz_density_per_row
                * max_nnz_density_per_row
                * std_dev_nnz_density_per_row
        """
        features = {
            "name": self.name
        }
        num_rows, num_cols = self.coo_mtx.shape
        features["num_rows"] = num_rows
        features["num_cols"] = num_cols
        features["nnz"] = self.coo_mtx.nnz

        # Collect nnz per row
        nnz_per_row = []
        indptr, _, _ = self.adj_tensors("csr")
        assert len(indptr) == num_rows + 1, F"len(indptr) {len(indptr)} is not equal to num_rows + 1 ({num_rows + 1})."
        for row_ind in range(num_rows):
            nnz_per_row.append(indptr[row_ind + 1] - indptr[row_ind])
        avg_nnz_per_row = np.mean(nnz_per_row)
        min_nnz_per_row = min(nnz_per_row)
        max_nnz_per_row = max(nnz_per_row)
        std_dev_nnz_per_row = np.std(nnz_per_row)
        avg_nnz_densitry_per_row = avg_nnz_per_row / num_cols
        min_nnz_densitry_per_row = min_nnz_per_row / num_cols
        max_nnz_densitry_per_row = max_nnz_per_row / num_cols
        std_dev_nnz_density_per_row = np.std([x / num_cols for x in nnz_per_row])
        features["avg_nnz_per_row"] = avg_nnz_per_row
        features["min_nnz_per_row"] = min_nnz_per_row
        features["max_nnz_per_row"] = max_nnz_per_row
        features["std_dev_nnz_per_row"] = std_dev_nnz_per_row
        features["avg_nnz_density_per_row"] = avg_nnz_densitry_per_row
        features["min_nnz_density_per_row"] = min_nnz_densitry_per_row
        features["max_nnz_density_per_row"] = max_nnz_densitry_per_row
        features["std_dev_nnz_density_per_row"] = std_dev_nnz_density_per_row

        return features


    def init_buckets(self, num_parts: int, width_limit: int=1024):
        # row_indices = [None] * num_parts
        # col_indices = [None] * num_parts

        num_rows = self.num_src_nodes()
        num_cols = self.num_dst_nodes()
        partition_size = (num_cols + num_parts - 1) // num_parts

        # # test
        # print(F"num_rows: {num_rows} num_cols: {num_cols} num_parts: {num_parts} partition_size: {partition_size}")
        # # end test

        # Count the degree of each row in each partition
        # degree_counter shape is partition_size * num_rows
        degree_counter = [ [0] * num_rows for i in range(num_parts) ]
        for row_ind, col_ind in zip(self.coo_mtx.row, self.coo_mtx.col):
            part_ind = col_ind // partition_size
            degree_counter[part_ind][row_ind] += 1
        
        # # test
        # for part_ind in range(num_parts):
        #     print(F"degree_counter[{part_ind}]: {degree_counter[part_ind]}")
        # # end test

        # Put rows into its corresponding bucket, a row with length l and
        # 2^{i - 1} < l <= 2^{i} should be in the bucket with width 2^{i}
        buckets = []
        for part_ind in range(num_parts):
            # buckets.append({})
            b_pool = {}
            for row_ind in range(num_rows):
                degree = degree_counter[part_ind][row_ind]
                # # test
                # print(F"degree_counter[{part_ind}][{row_ind}]: {degree}")
                # # end test
                if 0 == degree:
                    continue
                pow_ceil = math.ceil(math.log2(degree))
                width = int(2 ** pow_ceil)
                new_rows = 1
                if width > width_limit:
                    # Limit the width according to GPU block thread limit (1024 for CUDA)
                    ratio = width // width_limit
                    width = width_limit
                    new_rows *= ratio

                if width not in b_pool:
                    # A new bucket
                    b_pool[width] = Bucket(width, new_rows, degree)
                else:
                    # Existing bucket
                    b_pool[width].nnz += degree
                    b_pool[width].num_rows += new_rows
            b_pool = dict(sorted(b_pool.items())) # sort by keys
            buckets.append(b_pool)
        
        return buckets
                
