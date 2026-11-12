"""Reuse the existing painted-area converter with explicit observation/review gates."""

import numpy as np
from rasterio.features import rasterize

from roadmarking.dataset.labels import BACKGROUND, INSTANCE, NO_DATA, UNKNOWN, build_tile_labels


def reviewed_masks(affine, shape, observed, valid, features, completeness=None, uncertain=None):
    observed, valid = np.asarray(observed, bool), np.asarray(valid, bool)
    if observed.shape != shape or valid.shape != shape or (observed & ~valid).any():
        raise ValueError("Aligned source/observed masks required")
    labels = build_tile_labels(affine, shape, observed, features, completeness=completeness)
    uncertain_pixels = np.zeros(shape, bool)
    if uncertain is not None and not uncertain.is_empty:
        uncertain_pixels = (
            rasterize([(uncertain, 1)], out_shape=shape, transform=affine, dtype="uint8").astype(
                bool
            )
            & observed
            & valid
        )
    ignored = ~observed | ~valid | uncertain_pixels
    labels.instance[ignored] = 0
    labels.semantic[ignored] = 255
    labels.state[~observed | ~valid] = NO_DATA
    labels.state[uncertain_pixels] = UNKNOWN
    for record in labels.instances:
        if record["local_id"]:
            record["pixels"] = int((labels.instance == record["local_id"]).sum())
            record["observed_pixels"] = record["pixels"]
    categories = labels.state.copy()
    categories[~valid] = 6
    categories[uncertain_pixels] = 7
    return labels, categories


def processor_targets(rgb, labels, *, inspection_only=False):
    """Fail closed: stock mask losses have no per-pixel review-validity argument.

    Inspection mode demonstrates tensor format, never training readiness. Background
    has no object mask; it is negative space in each paint-instance target. Ontology
    IDs 1..9 are explicitly mapped to model target IDs 0..8; never change GIS IDs.
    """
    from transformers import Mask2FormerImageProcessor

    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*labels.instance.shape, 3):
        raise ValueError("Use clean uint8 replicated-grayscale RGB, rescaled exactly once")
    if not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(
        rgb[:, :, 1], rgb[:, :, 2]
    ):
        raise ValueError("Replicated grayscale required")
    supervised = (labels.state == BACKGROUND) | (labels.state == INSTANCE)
    if not supervised.all() and not inspection_only:
        raise ValueError("Stock Mask2Former cannot ignore unknown/NoData/boundary/padding pixels")
    segmentation = np.where(labels.state == INSTANCE, labels.instance, 0).astype(np.int64)
    mapping = {
        r["local_id"]: r["visual_id"] - 1 for r in labels.instances if r["local_id"] and r["pixels"]
    }
    processor = Mask2FormerImageProcessor(do_resize=False, ignore_index=0, do_reduce_labels=False)
    encoded = processor(
        images=rgb,
        segmentation_maps=segmentation,
        instance_id_to_semantic_id=mapping,
        return_tensors="pt",
    )
    if tuple(encoded["pixel_values"].shape[-2:]) != labels.instance.shape:
        raise ValueError("Processor changed spatial sampling")
    return encoded
