import numpy as np

from helpers import rotate_vectors as rv
from . import test_util

def test_one_rotate_vectors_once(vectors, euler_d, rot_seq, vectors_rotated_expected, on_frame):
  print('============================test rotate vectors once============================')

  vectors_rotated = rv.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, 1, on_frame)

  result = test_util.get_result(np.allclose(vectors_rotated, vectors_rotated_expected))
  print('***vetors rotated: %s***' % result)
  print('expected:\n%s' % vectors_rotated_expected)
  print('rotated:\n%s\n' % vectors_rotated)

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
  test_one_rotate_vectors_once(vectors, euler_d, 'zyx', vectors_rotated_expected, False)

  vectors_rotated_expected = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1]
  ])
  test_one_rotate_vectors_once(vectors, euler_d, 'zyx', vectors_rotated_expected, True)

  vectors_rotated_expected = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, -1]
  ])
  test_one_rotate_vectors_once(vectors, euler_d, 'ZYX', vectors_rotated_expected, False)

  vectors_rotated_expected = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
    [-1, 1, 1]
  ])
  test_one_rotate_vectors_once(vectors, euler_d, 'ZYX', vectors_rotated_expected, True)

def test_one_rotate_vectors_multple_times(vectors, euler_d, rot_seq, times, on_frame):
  print('============================test rotate vectors %s times============================' % times)

  vectors_rotated_one_by_one = vectors
  for i in range(1, times + 1):
    print('@%s time' % i)
    vectors_rotated_one_by_one = rv.rotate_vectors_by_euler(vectors_rotated_one_by_one, euler_d, rot_seq, True, 1, on_frame)

  vectors_rotated_composed = rv.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, times, on_frame)

  result = test_util.get_result(np.allclose(vectors_rotated_one_by_one, vectors_rotated_composed))
  print('***vectors rotated results are SAME between one by one and composed: %s***' % result)
  print('one by one:\n%s' % vectors_rotated_one_by_one)
  print('composed:\n%s\n' % vectors_rotated_composed)

  euler_d = euler_d * times
  vectors_rotated_multiply_angles = rv.rotate_vectors_by_euler(vectors, euler_d, rot_seq, True, 1, on_frame)

  # [ToDo] it maybe SAME if rotation on only one axis
  result = test_util.get_result(not np.allclose(vectors_rotated_one_by_one, vectors_rotated_multiply_angles))
  print('***vectors rotated results are DIFFERENT between composed and by multiply angles: %s***' % result)
  print('composed:\n%s' % vectors_rotated_composed)
  print('multiply angles:\n%s\n' % vectors_rotated_multiply_angles)

def test_rotate_vectors_multple_times():
  vectors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
  ])
  euler_d = np.array([3, 3, 3])

  test_one_rotate_vectors_multple_times(vectors, euler_d, 'zyx', 5, False)
  test_one_rotate_vectors_multple_times(vectors, euler_d, 'zyx', 5, True)
  test_one_rotate_vectors_multple_times(vectors, euler_d, 'ZYX', 5, False)
  test_one_rotate_vectors_multple_times(vectors, euler_d, 'ZYX', 5, True)

def test():
  test_rotate_vectors_once()
  test_rotate_vectors_multple_times()
