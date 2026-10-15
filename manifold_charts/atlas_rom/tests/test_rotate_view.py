import numpy as np

from atlas_rom.cases.rotate import CASE, VIEW_EMBED


def test_view_embed_returns_the_moving_b_link_origin():
    b_index = CASE.mechanism.body_order.index("B")
    q = np.zeros((2, 12 * len(CASE.mechanism.body_order)))
    q.reshape(-1, len(CASE.mechanism.body_order), 3, 4)[:, b_index, :, 3] = (
        (1.0, 2.0, 3.0), (-4.0, 5.0, -6.0))

    assert np.allclose(VIEW_EMBED(q), ((1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)))
