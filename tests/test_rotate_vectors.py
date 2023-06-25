import numpy as np
from numpy.testing import assert_allclose

from EasyEuler import rotate_vectors

def single_test_rotate_vectors_once_by_rot(vectors, rot, vectors_rotated_expected, on_frame):
  vectors_rotated = rotate_vectors.rotate_vectors(vectors, rot, 1, on_frame)
  assert_allclose(vectors_rotated, vectors_rotated_expected)

def single_test_rotate_vectors_once(vectors, euler_d, rot_seq, vectors_rotated_expected, on_frame):
  vectors_rotated = rotate_vectors.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, 1, on_frame)
  assert_allclose(vectors_rotated, vectors_rotated_expected, atol=1e-8)

def test_rotate_vectors_once():
  vectors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
  ])
  euler_d = np.array([45, 90, 45])

  vectors_rotated_expected = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 1]
  ])
  single_test_rotate_vectors_once(vectors, euler_d, 'zyx', vectors_rotated_expected, False)

  vectors_rotated_expected = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1]
  ])
  single_test_rotate_vectors_once(vectors, euler_d, 'zyx', vectors_rotated_expected, True)

  vectors_rotated_expected = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, -1]
  ])
  single_test_rotate_vectors_once(vectors, euler_d, 'ZYX', vectors_rotated_expected, False)

  vectors_rotated_expected = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
    [-1, 1, 1]
  ])
  single_test_rotate_vectors_once(vectors, euler_d, 'ZYX', vectors_rotated_expected, True)

def single_test_rotate_vectors_multple_times(vectors, euler_d, rot_seq, times, on_frame):
  vectors_rotated_one_by_one = vectors
  for i in range(1, times + 1):
    vectors_rotated_one_by_one = rotate_vectors.rotate_vectors_by_euler(vectors_rotated_one_by_one, euler_d, rot_seq, True, 1, on_frame)

  vectors_rotated_composed = rotate_vectors.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, times, on_frame)

  assert_allclose(vectors_rotated_one_by_one, vectors_rotated_composed)

  euler_d = euler_d * times
  vectors_rotated_multiply_angles = rotate_vectors.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, 1, on_frame)

  # [ToDo] it maybe SAME if rotation on only one axis
  assert not np.allclose(vectors_rotated_one_by_one, vectors_rotated_multiply_angles)

def test_rotate_vectors_multple_times():
  vectors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
  ])
  euler_d = np.array([3, 3, 3])

  single_test_rotate_vectors_multple_times(vectors, euler_d, 'zyx', 5, False)
  single_test_rotate_vectors_multple_times(vectors, euler_d, 'zyx', 5, True)
  single_test_rotate_vectors_multple_times(vectors, euler_d, 'ZYX', 5, False)
  single_test_rotate_vectors_multple_times(vectors, euler_d, 'ZYX', 5, True)
