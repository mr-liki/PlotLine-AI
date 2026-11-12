"""Native crop planning from supervision support, never from prediction performance."""

import numpy as np


def plan_native_crops(targets, valid, minimum_valid_fraction=0.1, minimum_target_pixels=4):
    if targets.shape[1:] != valid.shape or valid.ndim != 2 or valid.shape != (1024, 1024):
        raise ValueError("Aligned native 1024 region required")
    plans, covered, diagnostics = [], np.zeros_like(valid, bool), {}
    for size in (512, 384, 256):
        positions = sorted(set(range(0, 1024 - size + 1, size // 2)) | {1024 - size})
        eligible = 0
        for row in positions:
            for col in positions:
                v = valid[row : row + size, col : col + size]
                eligible += int(v.mean() >= minimum_valid_fraction)
        diagnostics[str(size)] = {
            "native_size": size,
            "eligible_context_windows": eligible,
            "minimum_valid_fraction": minimum_valid_fraction,
        }
    for row in (0, 256, 512):
        for col in (0, 256, 512):
            v = valid[row : row + 512, col : col + 512]
            if v.mean() >= minimum_valid_fraction:
                plans.append((row, col, 512))
                covered[row : row + 512, col : col + 512] |= v
    # Smaller crops only recover otherwise unsupported observed paint; never downsample.
    for target in targets:
        remaining = target & valid & ~covered
        if remaining.sum() < minimum_target_pixels:
            continue
        rows, cols = np.where(remaining)
        options = []
        for size in (384, 256):
            row = int(np.clip(np.median(rows) - size // 2, 0, 1024 - size))
            col = int(np.clip(np.median(cols) - size // 2, 0, 1024 - size))
            v = valid[row : row + size, col : col + size]
            amount = int(remaining[row : row + size, col : col + size].sum())
            if v.mean() >= minimum_valid_fraction and amount >= minimum_target_pixels:
                options.append((amount, size, row, col))
        if options:
            _, size, row, col = max(options)
            if (row, col, size) not in plans:
                plans.append((row, col, size))
                covered[row : row + size, col : col + size] |= valid[
                    row : row + size, col : col + size
                ]
    return plans, covered, diagnostics
