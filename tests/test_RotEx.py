import numpy as np
from scipy.spatial.transform import Rotation

from helpers import RotEx

from . import utils as test_utils
from . import test_rotate_vectors

def test_single_from_v1_2_v2(v1, v2, self_roll_angle):
  print('============================test single RotEx from v1 to v2============================')

  rot = RotEx.from_v1_2_v2(v1, v2, self_roll_angle, True)
  print('rot as euler(deg) in ZYX sequence: %s\n' % rot.as_euler('ZYX', True))

  print('test on v1 and v2')
  v2_expected = v2 / np.linalg.norm(v2) * np.linalg.norm(v1)
  test_rotate_vectors.test_single_rotate_vectors_once_by_rot(v1, rot, v2_expected, False)

  # test on a vector vertical with v1 and v2, which can reflect change on self_roll_angle
  print('test on vertical v1 and v2')
  vertical_v1 = RotEx.get_vertical_rotvec(v1, v2)
  self_roll_angle = np.deg2rad(self_roll_angle)
  rot_on_v2 = Rotation.from_rotvec(v2 / np.linalg.norm(v2) * self_roll_angle)
  vertical_v2 = rot_on_v2.apply(vertical_v1)
  test_rotate_vectors.test_single_rotate_vectors_once_by_rot(vertical_v1, rot, vertical_v2, False)

def test_from_v1_2_v2():
  v1 = np.array([0, 3, 0])
  v2 = np.array([2, 3, 4])
  self_roll_angle = 0
  test_single_from_v1_2_v2(v1, v2, self_roll_angle)

  v1 = np.array([1, 2, 3])
  v2 = np.array([4, 5, 6])
  self_roll_angle = 15
  test_single_from_v1_2_v2(v1, v2, self_roll_angle)

def test_single_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame):
  vectors_in_old_frame = test_utils.get_test_vectors()
  vectors_rotated_in_old_frame = rot_in_old_frame.apply(vectors_in_old_frame)

  vectors_in_new_frame = rot_old_2_new_frame.inv().apply(vectors_in_old_frame)
  vectors_rotated_in_new_frame_expected = rot_old_2_new_frame.inv().apply(vectors_rotated_in_old_frame)

  rot_in_new_frame = RotEx.get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame)
  vectors_rotated_in_new_frame = rot_in_new_frame.apply(vectors_in_new_frame)

  result = test_utils.get_result(np.allclose(vectors_rotated_in_new_frame, vectors_rotated_in_new_frame_expected))
  print('***vectors rotated in new frame: %s***' % result)
  print('result: %s' % vectors_rotated_in_new_frame)
  print('expected: %s\n' % vectors_rotated_in_new_frame_expected)

def test_get_rot_in_new_frame():
  rot_in_old_frame = Rotation.from_euler('ZYX', np.array([0, 0, 30]), True)
  rot_old_2_new_frame = Rotation.from_euler('ZYX', np.array([0, 60, 0]), True)
  test_single_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame)

  rot_in_old_frame = Rotation.from_euler('ZYX', np.array([10, 20, 30]), True)
  rot_old_2_new_frame = Rotation.from_euler('ZYX', np.array([40, 50, 60]), True)
  test_single_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame)

def test():
  test_from_v1_2_v2()
  test_get_rot_in_new_frame()
  #tests on other RotEx functions are covered in other test modules, like test_attitude.py
