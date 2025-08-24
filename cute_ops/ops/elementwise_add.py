import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import torch
from cutlass.cute.runtime import from_dlpack

# from cute_ops.utils import print_coords, print_tensor_or_layout_info


@cute.kernel
def elementwise_add_naive_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = tidx + bidx * bdim
    m, n = gA.shape
    mi, ni = thread_idx // n, thread_idx % n

    if mi < m and ni < n:
        a = gA[mi, ni]
        b = gB[mi, ni]
        c = a + b
        gC[mi, ni] = c


@cute.jit
def elementwise_add_naive(gA, gB, gC):
    thread_per_block = 256
    m, n = gA.shape

    grid = ((m * n) // thread_per_block + 1, 1, 1)
    block = (thread_per_block, 1, 1)

    elementwise_add_naive_kernel(gA, gB, gC).launch(grid=grid, block=block)


def run_elementwise_add_naive():
    M, N = 2048, 2048

    A = torch.rand((M, N), device="cuda", dtype=torch.float32)
    B = torch.rand((M, N), device="cuda", dtype=torch.float32)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    cute.compile(elementwise_add_naive, mA, mB, mC)(mA, mB, mC)
    torch.testing.assert_close(A + B, C, rtol=1e-3, atol=1e-3)


@cute.kernel
def elementwise_add_vectorize_load_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = tidx + bidx * bdim
    m, n = gA.shape[1]

    tile_m = thread_idx // n
    tile_n = thread_idx % n

    if tile_m < m and tile_n < n:
        a_tile = gA[(None, (tile_m, tile_n))].load()
        b_tile = gB[(None, (tile_m, tile_n))].load()
        gC[(None, (tile_m, tile_n))].store(a_tile + b_tile)


@cute.jit
def elementwise_add_vectorize_load(gA, gB, gC, copy_bits: cutlass.Constexpr = 128):
    num_threads_per_block = 256
    vector_size = copy_bits // gA.element_type.width

    gA = cute.zipped_divide(gA, (1, vector_size))  # ((1, vector_size), (m, n/vector_size))
    gB = cute.zipped_divide(gB, (1, vector_size))  # ((1, vector_size), (m, n/vector_size))
    gC = cute.zipped_divide(gC, (1, vector_size))  # ((1, vector_size), (m, n/vector_size))

    grid = (cute.size(gA, mode=[1]) // num_threads_per_block, 1, 1)
    block = (num_threads_per_block, 1, 1)

    elementwise_add_vectorize_load_kernel(gA, gB, gC).launch(grid=grid, block=block)


def run_elementwise_add_vectorize_load():
    M, N = 2048, 2048

    A = torch.rand((M, N), device="cuda", dtype=torch.float32)
    B = torch.rand((M, N), device="cuda", dtype=torch.float32)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    cute.compile(elementwise_add_vectorize_load, mA, mB, mC, copy_bits=128)(mA, mB, mC)
    torch.testing.assert_close(A + B, C, rtol=1e-3, atol=1e-3)


@cute.kernel
def elementwise_add_tv_layout_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    tiled_A = cute.composition(blkA, tv_layout)
    tiled_B = cute.composition(blkB, tv_layout)
    tiled_C = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, None)
    thrA = tiled_A[thr_coord]
    thrB = tiled_B[thr_coord]
    thrC = tiled_C[thr_coord]

    a_val = thrA.load()
    b_val = thrB.load()
    thrC.store(a_val + b_val)


@cute.jit
def elementwise_add_tv_layout(mA, mB, mC):
    thr_layout = cute.make_layout((16, 32), stride=(32, 1))
    value_layout = cute.make_layout((2, 8), stride=(8, 1))
    tile_mn, tv_layout = cute.make_layout_tv(thr_layout, value_layout)

    # print(f"thr_layout: {thr_layout}")
    # print(f"value_layout: {value_layout}")
    # print(f"tile_mn: {tile_mn}")
    # print(f"tv_layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tile_mn)  # ((TileM,TileN),(RestM,RestN))

    # print(f"gC: {gC}")
    # print(cute.size(gC, mode=[1]))
    # print(cute.size(tv_layout, mode=[0]))

    grid = (cute.size(gC, mode=[1]), 1, 1)
    block = (cute.size(tv_layout, mode=[0]), 1, 1)

    elementwise_add_tv_layout_kernel(gA, gB, gC, tv_layout).launch(grid=grid, block=block)


def run_elementwise_add_tv_layout():
    M, N = 1024, 1024

    A = torch.rand((M, N), device="cuda", dtype=torch.float32)
    B = torch.rand((M, N), device="cuda", dtype=torch.float32)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    cute.compile(elementwise_add_tv_layout, mA, mB, mC)(mA, mB, mC)
    torch.testing.assert_close(A + B, C, rtol=1e-3, atol=1e-3)


@cute.kernel
def elementwise_add_tv_layout_atom_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
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

    tiled_A = cute.composition(blkA, tv_layout)
    tiled_B = cute.composition(blkB, tv_layout)
    tiled_C = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, (None, None))
    thrA = tiled_A[thr_coord]
    thrB = tiled_B[thr_coord]
    thrC = tiled_C[thr_coord]

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    cute.copy(copy_atom_load, thrA, frgA)
    cute.copy(copy_atom_load, thrB, frgB)

    result = frgA.load() + frgB.load()
    frgC.store(result)
    cute.copy(copy_atom_store, frgC, thrC)


@cute.jit
def elementwise_add_tv_layout_atom(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    vector_size = copy_bits // mA.element_type.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tile_mn, tv_layout = cute.make_layout_tv(thr_layout, value_layout)

    # print_tensor_or_layout_info("thr_layout", thr_layout)
    # print_tensor_or_layout_info("value_layout", value_layout)
    # print_tensor_or_layout_info("tile_mn", tile_mn)
    # print_tensor_or_layout_info("tv_layout", tv_layout)
    # - thr_layout: Layout(shape=(4, 32), stride=(32, 1))
    # - value_layout: Layout(shape=(4, 4), stride=(4, 1))
    # - tile_mn: Unknown type <class 'tuple'> (16, 128)
    # - tv_layout: Layout(shape=((32, 4), (4, 4)), stride=((64, 4), (16, 1)))

    gA = cute.zipped_divide(mA, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tile_mn)  # ((TileM,TileN),(RestM,RestN))

    grid = (cute.size(gC, mode=[1]), 1, 1)
    block = (cute.size(tv_layout, mode=[0]), 1, 1)
    elementwise_add_tv_layout_atom_kernel(gA, gB, gC, tv_layout).launch(grid=grid, block=block)


def run_elementwise_add_tv_layout_atom():
    M, N = 1024, 1024

    A = torch.rand((M, N), device="cuda", dtype=torch.float32)
    B = torch.rand((M, N), device="cuda", dtype=torch.float32)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    mA = from_dlpack(A)
    mB = from_dlpack(B)
    mC = from_dlpack(C)

    cute.compile(elementwise_add_tv_layout_atom, mA, mB, mC)(mA, mB, mC)
    torch.testing.assert_close(A + B, C, rtol=1e-3, atol=1e-3)


@cute.kernel
def elementwise_add_copy_atom_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    C_coord: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    value_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)

    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkC_coord = C_coord[blk_coord]

    cute_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    cute_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(cute_atom_load, thr_layout, value_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(cute_atom_load, thr_layout, value_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(cute_atom_store, thr_layout, value_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)
    thrC_coord = thr_copy_C.partition_S(blkC_coord)

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)
    frgC_pred = cute.make_fragment_like(thrC_coord, dtype=cutlass.Boolean)

    for i in range(cute.size(frgC_pred), unrool_full=True):
        frgC_pred[i] = cute.elem_less(thrC_coord[i], shape)

    cute.copy(cute_atom_load, thrA, frgA, pred=frgC_pred)
    cute.copy(cute_atom_load, thrB, frgB, pred=frgC_pred)

    result = frgA.load() + frgB.load()
    frgC.store(result)

    cute.copy(cute_atom_store, frgC, thrC, pred=frgC_pred)


@cute.jit
def elementwise_add_copy_atom(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    vector_size = copy_bits // mA.element_type.width

    thr_layout = cute.make_ordered_layout(
        (8, 32), order=(1, 0)
    )  # 128 threads per block. 32 threads per warp. Total 4 warps.
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))  # 128 bits per value
    tile_mn, tv_layout = cute.make_layout_tv(thr_layout, value_layout)
    # print_tensor_or_layout_info("tile_mn", tile_mn)
    # print_tensor_or_layout_info("thr_layout", thr_layout)

    gA = cute.zipped_divide(mA, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tile_mn)  # ((TileM,TileN),(RestM,RestN))

    # Init gC coordinates to check if boundaries are correct
    idC = cute.make_identity_tensor(mC.shape)  # (M, N)
    # print_coords(idC)
    C_coord = cute.zipped_divide(idC, tile_mn)  # ((TileM,TileN),(RestM,RestN))
    grid = (cute.size(gC, mode=[1]), 1, 1)
    block = (cute.size(tv_layout, mode=[0]), 1, 1)
    elementwise_add_copy_atom_kernel(
        gA, gB, gC, C_coord, mC.shape, thr_layout, value_layout
    ).launch(grid=grid, block=block)


def run_elementwise_add_copy_atom():
    M, N = 1024, 1024

    A = torch.rand((M, N), device="cuda", dtype=torch.float32)
    B = torch.rand((M, N), device="cuda", dtype=torch.float32)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    cute.compile(elementwise_add_copy_atom, A, B, C)(A, B, C)
    torch.testing.assert_close(A + B, C, rtol=1e-3, atol=1e-3)


def benchmark():
    def generate_tensors():
        M, N = 12288, 12288
        A = torch.rand((M, N), device="cuda", dtype=torch.float32)
        B = torch.rand((M, N), device="cuda", dtype=torch.float32)
        C = torch.zeros((M, N), device="cuda", dtype=torch.float32)
        return testing.JitArguments(A, B, C)

    A, B, C = generate_tensors().args
    for func in [
        elementwise_add_naive,
        elementwise_add_vectorize_load,
        elementwise_add_tv_layout,
        elementwise_add_tv_layout_atom,
        elementwise_add_copy_atom,
    ]:
        # start = time.time()
        compiled_func = cute.compile(func, A, B, C)
        assert compiled_func is not None
        # end = time.time()
        # print(f"{func.__name__} compile time: {end - start} seconds")

        avg_time_us = testing.benchmark(
            compiled_func,
            workspace_generator=generate_tensors,
            workspace_count=10,
            warmup_iterations=2,
            iterations=100,
        )
        print(f"{func.__name__} Kernel execution time: {avg_time_us / 1e3:.4f} ms")


if __name__ == "__main__":
    # run_elementwise_add_naive()
    # run_elementwise_add_vectorize_load()
    # run_elementwise_add_tv_layout()
    # run_elementwise_add_tv_layout_atom()
    # run_elementwise_add_copy_atom()
    benchmark()
