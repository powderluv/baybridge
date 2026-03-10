import baybridge as bb


def test_make_composed_layout_with_identity_outer() -> None:
    def inner(coord):
        x, y = coord
        return x, y + 1

    layout = bb.make_composed_layout(inner, (1, 0), bb.make_identity_layout((8, 4)))

    assert layout.shape == (8, 4)
    assert layout(0) == (1, 1)
    assert layout((2, 3)) == (3, 4)


def test_make_composed_layout_can_gather_from_offset_tensor() -> None:
    offsets = bb.tensor([3, 1, 4, 1], dtype="index")

    def inner(coord):
        return offsets[coord]

    layout = bb.make_composed_layout(inner, 0, bb.make_layout((4,), stride=(1,)))

    assert [layout(index) for index in range(4)] == [3, 1, 4, 1]
