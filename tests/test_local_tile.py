import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


def print_local_tile():
    A = torch.rand(32, 64, dtype=torch.float32)
    mA = from_dlpack(A)

    tiler = (8, 4, 2)
    tiler_coord = (0, 0, None)

    print(f"{mA.shape=}")
    gA = cute.local_tile(mA, tiler, tiler_coord, proj=(1, 1, None))
    print(f"{gA.shape=}")


if __name__ == "__main__":
    print_local_tile()
