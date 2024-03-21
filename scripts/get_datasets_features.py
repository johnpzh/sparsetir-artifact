import dgl
import numpy as np
from utils import get_dataset

def print_edgelist(edgelist):
    for row_ind in range(len(edgelist)):
        row = edgelist[row_ind]
        for col_ind in row:
            print(F"({row_ind}, {col_ind})")

def get_basics(graph):
    csr_indptr, csr_indices, _ = graph.adj_tensors(fmt='csr')
    num_rows = len(csr_indptr) - 1
    assert num_rows == graph.num_nodes()
    num_edges = len(csr_indices)
    assert num_edges == graph.num_edges()

    edgelist = []
    for row_ind in range(num_rows):
        row = []
        for col_loc in range(csr_indptr[row_ind], csr_indptr[row_ind + 1]):
            col_ind = csr_indices[col_loc]
            row.append(col_ind)
        edgelist.append(row)
    
    return num_rows, num_edges, edgelist


datasets = [
    "arxiv",
    "proteins",
    "pubmed",
    "citeseer",
    "cora",
    "ppi",
    "reddit",
    "products",
]

if __name__ == "__main__":
    # for name in datasets:
    #     graph = get_dataset(name)

    # test
    g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    num_rows, num_edges, edgelist = get_basics(g)
    print_edgelist(edgelist)
    # end test

    # Statistics
    row_sizes = [len(row) for row in edgelist] # number of non-zeros of each row
    num_cols = num_rows
    nnz = num_edges
    avg_nnz_per_row = nnz / num_rows
    min_nnz_per_row = min(row_sizes)
    max_nnz_per_row = max(row_sizes)
    std_dev_nnz_per_row = np.std(row_sizes)

