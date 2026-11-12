"""Phase 4: annotation inspection, alignment gating and Mask2Former label preparation.

Nothing in this package reprojects, shifts, rotates or rescales client annotations.
Instance masks are only produced for features whose QC status is explicitly allowed.
"""
