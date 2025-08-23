import cutlass.cute as cute

__all__ = ["print_coords", "print_tensor_or_layout_info"]


def print_coords(tensor):
    shape = tensor.shape
    rank = len(shape)
    print(f"Tensor shape: {shape}, rank: {rank}")
    for idx in range(cute.size(tensor)):
        coord = tensor[idx]
        print(f"idx={idx}: coord={coord}")


def print_tensor_or_layout_info(name, obj):
    if isinstance(obj, cute.Tensor):
        print(
            f"{name}: Tensor(shape={obj.shape}, element_type={obj.element_type}, layout={obj.layout})"
        )
    elif isinstance(obj, cute.Layout):
        print(f"{name}: Layout(shape={obj.shape}, stride={obj.stride})")
    else:
        print(f"{name}: Unknown type {type(obj)} {obj}")
