import operator

import cuda.bindings.driver as cuda
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
    blkC_coord = C_coord[blk_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiledA = cute.composition(blkA, tv_layout)
    tiledB = cute.composition(blkB, tv_layout)
    tiledC = cute.composition(blkC, tv_layout)
    tiledC_coord = cute.composition(blkC_coord, tv_layout)

    # print("tiledA:", tiledA)
    # - tiledA: tensor<ptr<f32, gmem> o ((32,4),(4,4)):((4,4096),(1,1024))>

    thr_coord = (tidx, (None, None))
    thrA = tiledA[thr_coord]
    thrB = tiledB[thr_coord]
    thrC = tiledC[thr_coord]
    thrC_coord = tiledC_coord[thr_coord]

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)
    frgC_coord = cute.make_fragment_like(thrC_coord, dtype=cutlass.Boolean)

    for i in cutlass.range_constexpr(cute.size(frgC_coord)):
        frgC_coord[i] = cute.elem_less(thrC_coord[i], shape)

    cute.copy(copy_atom_load, thrA, frgA, pred=frgC_coord)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgC_coord)
    result = op(frgA.load(), frgB.load())
    frgC.store(result)
    cute.copy(copy_atom_store, frgC, thrC, pred=frgC_coord)


@cute.jit
def elementwise_apply(
    op: cutlass.Constexpr,
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: cuda.CUstream,
    copy_bits: cutlass.Constexpr = 128,
):
    vector_size = copy_bits // mA.element_type.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tile_mn, tv_layout = cute.make_layout_tv(thr_layout, value_layout)

    gA = cute.zipped_divide(mA, tile_mn)  # ((tile_m, tile_n), (rest_m, rest_n))
    gB = cute.zipped_divide(mB, tile_mn)  # ((tile_m, tile_n), (rest_m, rest_n))
    gC = cute.zipped_divide(mC, tile_mn)  # ((tile_m, tile_n), (rest_m, rest_n))

    shape = mC.shape
    idC = cute.make_identity_tensor(shape)
    C_coord = cute.zipped_divide(idC, tile_mn)

    grid = (cute.size(gC, mode=[1]), 1, 1)
    block = (cute.size(tv_layout, mode=[0]), 1, 1)
    elementwise_apply_kernel(op, gA, gB, gC, C_coord, shape, tv_layout).launch(
        grid=grid, block=block, stream=stream
    )


def run_elementwise_apply():
    M, N = 1024, 1024
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    B = torch.randn(M, N, device="cuda", dtype=torch.float32)
    C = torch.zeros_like(A)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    # Create non default CUDA stream from PyTorch
    torch_stream = torch.cuda.Stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    op = operator.add
    cute.compile(elementwise_apply, op, mA, mB, mC, current_stream)(mA, mB, mC, current_stream)
    torch.testing.assert_close(op(A, B), C)


if __name__ == "__main__":
    run_elementwise_apply()
