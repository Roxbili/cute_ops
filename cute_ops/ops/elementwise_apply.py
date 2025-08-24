import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elementwise_apply_kernel(
    op: cutlass.Constexpr,
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    C_coord: cute.Tensor,
    shape: cute.Shape,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiledA = cute.composition(blkA, tv_layout)
    tiledB = cute.composition(blkB, tv_layout)
    tiledC = cute.composition(blkC, tv_layout)

    print("tiledA:", tiledA)

    # thr_coord = ()
    # thr_A =


@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    copy_bits: cutlass.Constexpr = 128,
):
    vector_size = copy_bits // (8 * mA.element_size())

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tile_mn, tv_layout = cute.make_layout_tv(thr_layout, value_layout)

    gA = cute.zipped_divide(mA, tile_mn)  # ((tile_m, tile_n), (rest_m, rest_n))
    gB = cute.zipped_divide(mB, tile_mn)  # ((tile_m, tile_n), (rest_m, rest_n))
    gC = cute.zipped_divide(mC, tile_mn)  # ((tile_m, tile_n), (rest_m, rest_n))

    shape = mC.shape
    idC = cute.make_identity_tensor(shape)
    C_coord = cute.zipped_divide(idC, tile_mn)

    grid = (cute.size(gC, model=[1]), 1, 1)
    block = (cute.size(tv_layout, model=[0]), 1, 1)
    elementwise_apply_kernel(op, gA, gB, gC, C_coord, shape, tv_layout).launch(grid, block)


def run_elementwise_apply():
    M, N = 1024, 1024
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    B = torch.randn(M, N, device="cuda", dtype=torch.float32)
    C = torch.zeros_like(A)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    cute.compile(elementwise_apply, mA, mB, mC)(mA, mB, mC)


if __name__ == "__main__":
    run_elementwise_apply()
