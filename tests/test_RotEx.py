import numpy as np
from scipy.spatial.transform import Rotation

from helpers import RotEx

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

def test():
  test_from_v1_2_v2()
  #tests on other RotEx functions are covered in other test modules, like test_attitude.py
