import sys
from utils import get_dataset

def print_edgelist(edgelist):
    for row_ind in range(len(edgelist)):
        row = edgelist[row_ind]
        for col_ind in row:
            print(F"({row_ind}, {col_ind})")


def get_basics(graph):
    csr_indptr, csr_indices, _ = graph.adj_sparse(fmt='csr')
    num_rows = len(csr_indptr) - 1
    assert num_rows == graph.num_nodes()
    num_edges = len(csr_indices)
    assert num_edges == graph.num_edges()

    print(F"num_rows: {num_rows} num_edges: {num_edges}", flush=True)
    edgelist = []
    for row_ind in range(num_rows):
        row = []
        for col_loc in range(csr_indptr[row_ind], csr_indptr[row_ind + 1]):
            col_ind = csr_indices[col_loc]
            row.append(col_ind)
        row.sort()
        edgelist.append(row)
    
    return num_rows, num_edges, edgelist


def save_to_mtx_v1(num_rows,
                num_edges,
                edgelist,
                filename):
    with open(filename, "w") as fout:
        fout.write("%%MatrixMarket matrix coordinate real general\n")
        num_cols = num_rows
        fout.write(F"{num_rows} {num_cols} {num_edges}\n")
        for row_ind in range(num_rows):
            row = edgelist[row_ind]
            if len(row) == 0:
                continue
            for col_ind in row:
                val = 17
                fout.write(F"{row_ind + 1} {col_ind + 1} {val}\n")

def save_to_mtx_v2(graph,
                   filename):
    csr_indptr, csr_indices, _ = graph.adj_sparse(fmt='csr')
    num_rows = len(csr_indptr) - 1
    assert num_rows == graph.num_nodes()
    num_edges = len(csr_indices)
    assert num_edges == graph.num_edges()

    print(F"num_rows: {num_rows}")
    print(F"num_edges: {num_edges}")
    print(F"Saving to {out_filename} ...", flush=True)

    with open(filename, "w") as fout:
        fout.write("%%MatrixMarket matrix coordinate real general\n")
        num_cols = num_rows
        fout.write(F"{num_rows} {num_cols} {num_edges}\n")
        for row_ind in range(num_rows):
            for col_loc in range(csr_indptr[row_ind], csr_indptr[row_ind + 1]):
                col_ind = csr_indices[col_loc]
                val = 17
                fout.write(F"{row_ind + 1} {col_ind + 1} {val}\n")
            if row_ind % 10000 == 0:
                print(".", end="", flush=True)
        print("")



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
    if len(sys.argv) < 3:
        raise RuntimeError(F"Usage: {sys.argv[0]} <dataset_name> <output.mtx>")
    
    dataset_name = sys.argv[1]
    out_filename = sys.argv[2]

    g = get_dataset(dataset_name)
    # num_rows, num_edges, edgelist = get_basics(g)
    # # test
    # g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    # num_rows, num_edges, edgelist = get_basics(g)
    # print_edgelist(edgelist)
    # # end test


    print(F"dataset: {dataset_name}")
    save_to_mtx_v2(g, out_filename)

    # print(F"num_rows: {num_rows}")
    # print(F"num_edges: {num_edges}")
    # print(F"Saving to {out_filename} ...", flush=True)
    # save_to_mtx(num_rows,
    #             num_edges,
    #             edgelist,
    #             out_filename)
    print("Done.")

