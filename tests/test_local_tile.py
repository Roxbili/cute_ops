import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.jit
def print_local_tile_jit(mA):
    tiler = (8, 4, 2)
    tiler_coord = (0, 0, None)
    cute.printf("mA shape: ")
    cute.printf(mA.shape)

    gA = cute.local_tile(mA, tiler, tiler_coord, proj=(1, 1, None))
    cute.printf("gA shape: ")
    cute.printf(gA.shape)


def run():
    A = torch.rand(32, 64, dtype=torch.float32)
    mA = from_dlpack(A, assumed_align=16)
    print_local_tile_jit(mA)


if __name__ == "__main__":
    run()
