import numpy as np

from atlas_rom.cases.jansen_walker import CASE, VIEW_EMBED


def test_view_embed_places_the_master_crank_on_a_unit_circle():
    crank_index = CASE.mechanism.body_order.index("crank@R00")
    q = np.zeros((2, 12 * len(CASE.mechanism.body_order)))
    for sample, angle in enumerate((0.0, np.pi / 2.0)):
        rotation = np.array(((np.cos(angle), -np.sin(angle), 0.0),
                             (np.sin(angle), np.cos(angle), 0.0),
                             (0.0, 0.0, 1.0)))
        q[sample].reshape(-1, 3, 4)[crank_index, :, :3] = rotation

    points = VIEW_EMBED(q)

    assert np.allclose(points, ((1.0, 0.0), (0.0, 1.0)))
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
