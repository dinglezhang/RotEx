import numpy as np

from helpers import vectors_rotation as vr

def test_vectors_rotation_once(is_extrinsic):
  way_str = 'intrinsicly'
  if is_extrinsic:
    way_str = 'extrinsicly'
  print('============================test vectors rotation %s once============================' % way_str)

  vectors_input = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
  ])
  euler_d_input = np.array([45, 90, 45])
  vr.vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, False)
  vr.vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, True)

def test_vectors_rotation_multple_times(is_extrinsic):
  way_str = 'intrinsicly'
  if is_extrinsic:
    way_str = 'extrinsicly'
  print('============================test vectors rotation %s 5 times============================' % way_str)

  vectors_input = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
  ])
  euler_d_input = np.array([3, 3, 3])
  times = 5

  vectors_rotated_one_by_one = vectors_input
  for i in range(1, times + 1):
    print('@' + str(i))
    vectors_rotated_one_by_one = vr.vectors_rotation(vectors_rotated_one_by_one, euler_d_input, 'zyx', 1, is_extrinsic, False)

  vectors_rotated_composed = vr.vectors_rotation(vectors_input, euler_d_input, 'zyx', times, is_extrinsic, False)

  result = vr.common.get_result(np.allclose(vectors_rotated_one_by_one, vectors_rotated_composed))
  print('***vectors rotated results are SAME between one by one and composed: %s***' % result)
  print('one by one:\n%s' % vectors_rotated_one_by_one)
  print('composed:\n%s\n' % vectors_rotated_composed)

  euler_d_input = euler_d_input * times
  vectors_rotated_multiple_angles = vr.vectors_rotation(vectors_input, euler_d_input, 'zyx', 1, is_extrinsic, False)

  # [ToDo] it maybe SAME if rotation on only one axis
  result = vr.common.get_result(not np.allclose(vectors_rotated_one_by_one, vectors_rotated_multiple_angles))
  print('***vectors rotated results are DIFFERENT between one by one and by multiple angles: %s***' % result)
  print('one by one:\n%s' % vectors_rotated_one_by_one)
  print('multiple angles:\n%s\n' % vectors_rotated_multiple_angles)

def test():
  test_vectors_rotation_once(False)
  test_vectors_rotation_multple_times(False)

  test_vectors_rotation_once(True)
  test_vectors_rotation_multple_times(True)
