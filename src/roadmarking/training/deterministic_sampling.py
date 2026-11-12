"""Deterministic replacement for the criterion's bilinear point sampling (Stage 9E-I).

``F.grid_sample`` backward on CUDA uses atomic adds and is nondeterministic; it was the
only nondeterministic op PyTorch reported in Stage 9E. This mixin reproduces
``grid_sample(mode="bilinear", padding_mode="border", align_corners=False)`` at native
pixel centres with explicit four-neighbour gathers, whose backward has a deterministic
implementation under ``torch.use_deterministic_algorithms(True)``.
"""


class ExactBilinearMixin:
    def logits_at(self, predictions, rows, cols, height, width):
        if not self.bilinear:
            return super().logits_at(predictions, rows, cols, height, width)
        ph, pw = predictions.shape[-2:]
        y = ((rows.double() + 0.5) * ph / height - 0.5).clamp(0, ph - 1)
        x = ((cols.double() + 0.5) * pw / width - 0.5).clamp(0, pw - 1)
        y0, x0 = y.floor().long(), x.floor().long()
        y1, x1 = (y0 + 1).clamp(max=ph - 1), (x0 + 1).clamp(max=pw - 1)
        wy, wx = (y - y0).float(), (x - x0).float()
        p = predictions.float()
        return (
            p[:, y0, x0] * (1 - wx) * (1 - wy)
            + p[:, y0, x1] * wx * (1 - wy)
            + p[:, y1, x0] * (1 - wx) * wy
            + p[:, y1, x1] * wx * wy
        )


def with_exact_bilinear(criterion_class):
    return type(f"Exact{criterion_class.__name__}", (ExactBilinearMixin, criterion_class), {})
