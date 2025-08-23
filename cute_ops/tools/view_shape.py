import matplotlib.pyplot as plt


def visualize_tiles(tile_shape, layout_shape):
    """可视化二维 tile 布局（行列方式），即使有单行/单列也按二维画
    Args:
        tile_shape:  (tile_h, tile_w)  每个tile的行列大小
        layout_shape: (tile_rows, tile_cols) tile的排布行列

    例如 tile_shape=(4,6), layout_shape=(1,2) 表示每个 tile 是 4 行 6 列，排成 1 行 2 列
    """
    tile_h, tile_w = tile_shape
    tile_rows, tile_cols = layout_shape
    total_h = tile_h * tile_rows
    total_w = tile_w * tile_cols

    fig, ax = plt.subplots(figsize=(max(5, total_w / 2), max(5, total_h / 2)))

    # 背景细网格（单元格）
    for i in range(total_h + 1):
        ax.axhline(i, color="lightgray", linewidth=0.5)
    for j in range(total_w + 1):
        ax.axvline(j, color="lightgray", linewidth=0.5)

    # tile 边界
    for r in range(tile_rows + 1):
        ax.axhline(r * tile_h, color="red", linewidth=2)
    for c in range(tile_cols + 1):
        ax.axvline(c * tile_w, color="red", linewidth=2)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(range(total_w + 1))
    ax.set_yticks(range(total_h + 1))
    ax.set_title(f"tile size={tile_shape}, layout={layout_shape}, total=({total_h},{total_w})")
    plt.show()
