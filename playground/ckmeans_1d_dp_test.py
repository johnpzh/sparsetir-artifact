import matplotlib.cm as cm
from scipy.io import mmread
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ckmeans_1d_dp import ckmeans
import statistics
from itertools import accumulate
import operator


import math

# Function to find the nearest power
# of two for every array element
def nearestPowerOfTwo(arr, N):
    # Traverse the array
    for i in range(N):
        # Calculate log of current array element
        lg = (int)(math.log2(arr[i]))
 
        a = (int)(math.pow(2, lg))
        b = (int)(math.pow(2, lg + 1))
 
        # Find the nearest
        if ((arr[i] - a) < (b - arr[i])):
            print(a, end = " ")
        else:
            print(b, end = " ")


def find_cluster_indices(lst, condition):
   return [i for i, elem in enumerate(lst) if condition(elem)]

def cumulative_sum(input_list):
    # Use the accumulate() function to perform a cumulative sum of the elements in the list
    cumulative_sum_iter = accumulate(input_list, operator.add)
    # Convert the iterator to a list and return it
    return list(cumulative_sum_iter)

inputs = ["ins2", "ASIC_680k", "eu-2005", "web-BerkStan"]
# inputs = ["ins2"]

for input in inputs:
    abs_sparse_file_path = f"./../data/suitesparse/{input}/{input}.mtx"
    # abs_sparse_file_path = "./"  + input + ".mtx"
    print("\nInputFile: ", input)
    # Replace 'file.mtx' with the path to your Matrix Market file
    sparse_matrix = mmread(abs_sparse_file_path)
    # Assume 'data' is your sparse matrix
    sparse_matrix_csr = csr_matrix(sparse_matrix)
    X = sparse_matrix.getnnz(axis=1)

    cluster_ids = ckmeans(X).cluster
    num_cluster = max(cluster_ids + 1)  

    stats = []
    for i in range(num_cluster):
        splited_cluster_ids = find_cluster_indices(cluster_ids, lambda e: e == i)
        cluster_nnz_values =  X[splited_cluster_ids]
        # print(cluster_nnz_values)
        std_value = 0
        if len(cluster_nnz_values) > 1:
            std_value = statistics.stdev(cluster_nnz_values)

        stats.append([
                       len(cluster_nnz_values), 
                        min(cluster_nnz_values), 
                        max(cluster_nnz_values), 
                        statistics.mean(cluster_nnz_values),
                        std_value
                        ])

    nnz_each_cluster =  list(list(zip(*stats))[0])
    accum_nnz = cumulative_sum(nnz_each_cluster)
    max_nnz = accum_nnz[-1]
    #print([i/max_nnz*100 for i in accum_nnz])
    print(stats)
    


