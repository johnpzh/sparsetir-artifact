import argparse
import sys
from format_matrix_market import MTX

if __name__ == "__main__":
    parser = argparse.ArgumentParser("print mtx matrix in dense format")
    parser.add_argument("--dataset", "-d", type=str, help="matrix market (mtx) dataset path")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    filename = args.dataset
    g = MTX(filename)

    print(F"Print {filename}...")
    print(F"{g.coo_mtx.toarray()}")
    # print(F"{g.coo_mtx.todense()}")

