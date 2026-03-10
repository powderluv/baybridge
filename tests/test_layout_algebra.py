import baybridge as bb


def test_layout_accepts_scalar_shape_and_maps_linear_indices() -> None:
    layout = bb.make_layout(4, stride=2)

    assert layout.shape == (4,)
    assert layout.stride == (2,)
    assert layout(0) == 0
    assert layout(3) == 6


def test_coalesce_merges_contiguous_row_major_layout() -> None:
    layout = bb.make_layout((2, 3, 4), stride=(12, 4, 1))
    result = bb.coalesce(layout)

    assert result.shape == (24,)
    assert result.stride == (1,)
    assert bb.depth(result) == 1


def test_coalesce_preserves_non_contiguous_axes() -> None:
    layout = bb.make_layout((2, 3), stride=(7, 2))
    result = bb.coalesce(layout)

    assert result.shape == (2, 3)
    assert result.stride == (7, 2)


def test_flat_divide_computes_tile_and_rest_layout() -> None:
    layout = bb.make_layout((8, 6), stride=(6, 1))
    result = bb.flat_divide(layout, tiler=(2, 3))

    assert result.shape == (2, 3, 4, 2)
    assert result.stride == (6, 1, 12, 3)


def test_blocked_and_raked_product_return_expected_shapes() -> None:
    layout = bb.make_layout((2, 5), stride=(5, 1))
    tiler = bb.make_layout((3, 4), stride=(1, 3))

    blocked = bb.blocked_product(layout, tiler=tiler)
    raked = bb.raked_product(layout, tiler=tiler)

    assert blocked.shape == (2, 5, 3, 4)
    assert blocked.stride == (60, 12, 1, 3)
    assert raked.shape == (2, 3, 5, 4)
    assert raked.stride == (60, 1, 12, 3)


def test_logical_and_tiled_divide_build_hierarchical_layouts() -> None:
    layout = bb.make_layout((9, 4, 8), stride=(59, 13, 1))
    tiler = (
        bb.make_layout(3, stride=3),
        bb.make_layout((2, 4), stride=(1, 8)),
    )

    logical = bb.logical_divide(layout, tiler=tiler)
    zipped = bb.zipped_divide(layout, tiler=tiler)
    tiled = bb.tiled_divide(layout, tiler=tiler)
    flat = bb.flat_divide(layout, tiler=tiler)

    assert logical.shape == ((3, 3), (2, 2), (4, 2))
    assert logical.stride == ((59, 177), (13, 26), (1, 4))
    assert zipped.shape == ((3, 2, 4), (3, 2, 2))
    assert zipped.stride == ((59, 13, 1), (177, 26, 4))
    assert tiled.shape == ((3, 2, 4), 3, 2, 2)
    assert tiled.stride == ((59, 13, 1), 177, 26, 4)
    assert flat.shape == (3, 2, 4, 3, 2, 2)
    assert flat.stride == (59, 13, 1, 177, 26, 4)
    assert bb.depth(logical) == 2
    assert bb.size(zipped) == bb.size(layout)


def test_logical_zipped_tiled_and_flat_product_build_expected_shapes() -> None:
    layout = bb.make_layout((2, 5), stride=(5, 1))
    tiler = bb.make_layout((3, 4), stride=(1, 3))

    logical = bb.logical_product(layout, tiler=tiler)
    zipped = bb.zipped_product(layout, tiler=tiler)
    tiled = bb.tiled_product(layout, tiler=tiler)
    flat = bb.flat_product(layout, tiler=tiler)

    assert logical.shape == ((2, 5), (3, 4))
    assert logical.stride == ((5, 1), (10, 30))
    assert zipped.shape == ((2, 5), (3, 4))
    assert zipped.stride == ((5, 1), (10, 30))
    assert tiled.shape == ((2, 5), 3, 4)
    assert tiled.stride == ((5, 1), 10, 30)
    assert flat.shape == (2, 5, 3, 4)
    assert flat.stride == (5, 1, 10, 30)
    assert bb.size(flat) == bb.size(layout) * bb.size(tiler)


def test_make_identity_layout_matches_row_major() -> None:
    layout = bb.make_identity_layout((3, 4))

    assert layout.shape == (3, 4)
    assert layout(0) == (0, 0)
    assert layout(5) == (1, 1)
