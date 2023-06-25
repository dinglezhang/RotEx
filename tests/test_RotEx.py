import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from EasyEuler import RotEx

from . import test_rotate_vectors

def single_test_from_v1_2_v2(v1, v2, self_roll_angle):
  rot = RotEx.from_v1_2_v2(v1, v2, self_roll_angle, True)

  # test on v1 and v2
  v2_expected = v2 / np.linalg.norm(v2) * np.linalg.norm(v1)
  test_rotate_vectors.single_test_rotate_vectors_once_by_rot(v1, rot, v2_expected, False)

  # test on a vector vertical with v1 and v2, which can reflect change on self_roll_angle
  vertical_v1 = RotEx.get_vertical_rotvec(v1, v2)
  self_roll_angle = np.deg2rad(self_roll_angle)
  rot_on_v2 = Rotation.from_rotvec(v2 / np.linalg.norm(v2) * self_roll_angle)
  vertical_v2 = rot_on_v2.apply(vertical_v1)
  test_rotate_vectors.single_test_rotate_vectors_once_by_rot(vertical_v1, rot, vertical_v2, False)

def test_from_v1_2_v2():
  v1 = np.array([0, 3, 0])
  v2 = np.array([2, 3, 4])
  self_roll_angle = 0
  single_test_from_v1_2_v2(v1, v2, self_roll_angle)

  v1 = np.array([1, 2, 3])
  v2 = np.array([4, 5, 6])
  self_roll_angle = 15
  single_test_from_v1_2_v2(v1, v2, self_roll_angle)

def single_test_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame, vector_samples):
  vectors_in_old_frame = vector_samples
  vectors_rotated_in_old_frame = rot_in_old_frame.apply(vectors_in_old_frame)

  vectors_in_new_frame = rot_old_2_new_frame.inv().apply(vectors_in_old_frame)
  vectors_rotated_in_new_frame_expected = rot_old_2_new_frame.inv().apply(vectors_rotated_in_old_frame)

  rot_in_new_frame = RotEx.get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame)
  vectors_rotated_in_new_frame = rot_in_new_frame.apply(vectors_in_new_frame)

  assert_allclose(vectors_rotated_in_new_frame, vectors_rotated_in_new_frame_expected)

def test_get_rot_in_new_frame(vector_samples):
  rot_in_old_frame = Rotation.from_euler('ZYX', np.array([0, 0, 30]), True)
  rot_old_2_new_frame = Rotation.from_euler('ZYX', np.array([0, 60, 0]), True)
  single_test_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame, vector_samples)

  rot_in_old_frame = Rotation.from_euler('ZYX', np.array([10, 20, 30]), True)
  rot_old_2_new_frame = Rotation.from_euler('ZYX', np.array([40, 50, 60]), True)
  single_test_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame, vector_samples)

#tests on other RotEx functions are covered in other test modules, like test_attitude.py
