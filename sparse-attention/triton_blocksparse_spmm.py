import pytest
import torch
import argparse
import triton
from torch.profiler import profile, ProfilerActivity, schedule
from utils import create_pixelfly, create_longformer


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event)


def test_matmul(
    MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, sp_pattern, Z=1, H=1, M=512, N=384, K=256
):
    seed = 0
    torch.manual_seed(seed)
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
    # create inputs
    # create op
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]
    if sp_pattern == "pixelfly":
        layout = create_pixelfly(H, shape[0] // BLOCK, fmt="mask")
    elif sp_pattern == "longformer":
        layout = create_longformer(H, shape[0] // BLOCK, 256 // BLOCK, fmt="mask")
    else:
        raise KeyError("Sparse pattern {} not recongized.".format(args.pattern))

    # create data
    a_ref, a_tri = triton.testing.make_pair(a_shape, dtype=DTYPE, alpha=0.1)
    b_ref, b_tri = triton.testing.make_pair(b_shape, dtype=DTYPE, alpha=0.1)
    # compute [torch]
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    c_ref = torch.matmul(
        a_ref.transpose(2, 3) if TRANS_A else a_ref,
        b_ref.transpose(2, 3) if TRANS_B else b_ref,
    )
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    # triton result
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    op = triton.ops.blocksparse.matmul(
        layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda"
    )
    c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)
    # compare
    triton.testing.assert_almost_equal(c_ref, c_tri)

    print(
        "triton-block-spmm\tM\t{}\tN\t{}\tK\t{}\tblock\t{}\tdtype\t{}\ttrans_A\t{}\ttrans_B\t{}".format(
            M, N, K, BLOCK, DTYPE, TRANS_A, TRANS_B
        )
    )

    with profile(
        activities=[ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=10, active=100),
    ) as prof:
        for _ in range(100):
            op(a_tri, b_tri)
            prof.step()

    measure = sum([e.cuda_time for e in prof.events()]) / 1000 / 90
    print("triton time: \t{:.5f}ms".format(measure))
    return measure


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Triton sparse attention spmm")
    parser.add_argument(
        "--pattern", "-p", type=str, help="Sparse pattern: longformer/pixelfly"
    )
    args = parser.parse_args()
    test_matmul(
        "dsd",
        False,
        False,
        16,
        torch.float16,
        args.pattern,
        Z=1,
        H=12,
        M=4096,
        N=64,
        K=4096,
    )