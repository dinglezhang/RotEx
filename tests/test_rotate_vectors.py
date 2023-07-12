import pytest
import numpy as np
from numpy.testing import assert_allclose

from RotEx import rotate_vectors

def single_test_rotate_vectors_once_by_rot(vectors, rot, vectors_rotated_expected, on_frame):
  vectors_rotated = rotate_vectors.rotate_vectors(vectors, rot, 1, on_frame)
  assert_allclose(vectors_rotated, vectors_rotated_expected)

@pytest.mark.parametrize('vectors',
                        [np.array([
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1],
                                  [1, 1, 1]
                                ])])
@pytest.mark.parametrize('euler_d', [np.array([45, 90, 45])])
@pytest.mark.parametrize('rot_seq, vectors_rotated_expected, on_frame',
                         [('zyx', np.array([
                                  [0, 1, 0],
                                  [0, 0, 1],
                                  [1, 0, 0],
                                  [1, 1, 1]
                                ]), False),
                          ('zyx', np.array([
                                  [0, 0, 1],
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [1, 1, 1]
                                ]), True),
                          ('ZYX', np.array([
                                  [0, 0, -1],
                                  [0, 1, 0],
                                  [1, 0, 0],
                                  [1, 1, -1]
                                ]), False),
                          ('ZYX', np.array([
                                  [0, 0, 1],
                                  [0, 1, 0],
                                  [-1, 0, 0],
                                  [-1, 1, 1]
                                ]), True)])
def test_rotate_vectors_once(vectors, euler_d, rot_seq, vectors_rotated_expected, on_frame):
  vectors_rotated = rotate_vectors.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, 1, on_frame)
  assert_allclose(vectors_rotated, vectors_rotated_expected, atol = 1e-8)

@pytest.mark.parametrize('euler_d', [np.array([3, 3, 3])])
@pytest.mark.parametrize('rot_seq', ['ZYX', 'zyx'])
@pytest.mark.parametrize('times', [3, 4, 5])
@pytest.mark.parametrize('on_frame', [True, False])
def test_rotate_vectors_multple_times(vector_samples, euler_d, rot_seq, times, on_frame):
  vectors_rotated_one_by_one = vector_samples
  for i in range(1, times + 1):
    vectors_rotated_one_by_one = rotate_vectors.rotate_vectors_by_euler(vectors_rotated_one_by_one, euler_d, rot_seq, True, 1, on_frame)

  vectors_rotated_composed = rotate_vectors.rotate_vectors_by_euler(vector_samples, euler_d, rot_seq, True, times, on_frame)

  assert_allclose(vectors_rotated_one_by_one, vectors_rotated_composed)

  euler_d = euler_d * times
  vectors_rotated_multiply_angles = rotate_vectors.rotate_vectors_by_euler(vector_samples, euler_d, rot_seq, True, 1, on_frame)

  # [ToDo] it maybe SAME if rotation on only one axis
  assert not np.allclose(vectors_rotated_one_by_one, vectors_rotated_multiply_angles)
