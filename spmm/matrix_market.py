from os.path import basename, splitext
from scipy.io import mmread
import numpy as np

__all__ = ["MTX", "matrix_features"]

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
        features["avg_nnz_densitry_per_row"] = avg_nnz_densitry_per_row
        features["min_nnz_densitry_per_row"] = min_nnz_densitry_per_row
        features["max_nnz_densitry_per_row"] = max_nnz_densitry_per_row
        features["std_dev_nnz_density_per_row"] = std_dev_nnz_density_per_row

        return features

        

    
