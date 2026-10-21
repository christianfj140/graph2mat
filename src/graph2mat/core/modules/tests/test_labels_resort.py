import numpy as np

from graph2mat.core.modules._labels_resort import get_labels_resorting_array


def test_resorting_with_two_block_types_is_a_permutation():
    # Types: 0 = H (1x1 block), 1 = O (2x2 block); target order H, O, H.
    types = np.array([0, 1, 0], dtype=np.int64)
    shapes = np.array([[1, 2], [1, 2]])
    indices = np.asarray(get_labels_resorting_array(types, shapes=shapes))
    # Labels come concatenated per type: [H1, H2, O, O, O, O].
    assert indices.tolist() == [0, 2, 3, 4, 5, 1]
