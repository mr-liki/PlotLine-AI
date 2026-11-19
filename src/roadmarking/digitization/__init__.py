"""Road-marking digitization: accepted recognition -> source-aligned 3D vector features.

Modules: ``raster_tiles`` (production rasterization on a tile grid), ``recognition``
(registered model, frozen V1 contract, production stitching), ``fragments`` (observed
fragments, identity layers, continuity relations), ``centerline`` (skeleton graph,
sub-pixel centerlines), ``surface`` (robust LiDAR road-surface Z), ``export`` (GeoPackage),
``qa`` (geometry, identity, 3D and reference checks).
"""
