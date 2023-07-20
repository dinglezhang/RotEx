import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from RotEx import rotex

from . import test_rotate_vectors

rot_samples = \
  [Rotation.from_euler('ZYX', np.array([1, 2, 3]), True),
   Rotation.from_euler('ZYX', np.array([-1, 2, -3]), True),
   Rotation.from_euler('ZYX', np.array([3, -2, 10]), True),
   Rotation.from_euler('ZYX', np.array([-10, -1, 3]), True)]

rot_in_small_angles_samples = \
  [Rotation.from_euler('ZYX', np.array([0.1, 0.2, 0.3]), True),
   Rotation.from_euler('ZYX', np.array([-0.1, 0.2, -0.3]), True),
   Rotation.from_euler('ZYX', np.array([0.3, -0.2, 0.1]), True),
   Rotation.from_euler('ZYX', np.array([-0.2, -0.1, 0.3]), True)]

@pytest.mark.parametrize('v1', [np.array([0, 3, 0]), np.array([1, 2, 3])])
@pytest.mark.parametrize('v2', [np.array([2, 3, 4]), np.array([4, 5, 6])])
@pytest.mark.parametrize('self_roll_angle', [0, 15, 60, 90])
def test_from_v1_2_v2(v1, v2, self_roll_angle):
  rot = rotex.from_v1_2_v2(v1, v2, self_roll_angle, True)

  # test on v1 and v2
  v2_expected = v2 / np.linalg.norm(v2) * np.linalg.norm(v1)
  test_rotate_vectors.single_test_rotate_vectors_once_by_rot(v1, rot, v2_expected, False)

  # test on a vector vertical with v1 and v2, which can reflect change on self_roll_angle
  vertical_v1 = rotex.get_vertical_rotvec(v1, v2)
  self_roll_angle = np.deg2rad(self_roll_angle)
  rot_on_v2 = Rotation.from_rotvec(v2 / np.linalg.norm(v2) * self_roll_angle)
  vertical_v2 = rot_on_v2.apply(vertical_v1)
  test_rotate_vectors.single_test_rotate_vectors_once_by_rot(vertical_v1, rot, vertical_v2, False)

@pytest.mark.parametrize('rot_in_old_frame',
                        [Rotation.from_euler('ZYX', np.array([0, 0, 30]), True),
                         Rotation.from_euler('ZYX', np.array([10, 20, 30]), True)])
@pytest.mark.parametrize('rot_old_2_new_frame',
                        [Rotation.from_euler('ZYX', np.array([0, 60, 0]), True),
                         Rotation.from_euler('ZYX', np.array([40, 50, 60]), True)])
def test_get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame, vector_samples):
  vectors_in_old_frame = vector_samples
  vectors_rotated_in_old_frame = rot_in_old_frame.apply(vectors_in_old_frame)

  vectors_in_new_frame = rot_old_2_new_frame.inv().apply(vectors_in_old_frame)
  vectors_rotated_in_new_frame_expected = rot_old_2_new_frame.inv().apply(vectors_rotated_in_old_frame)

  rot_in_new_frame = rotex.get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame)
  vectors_rotated_in_new_frame = rot_in_new_frame.apply(vectors_in_new_frame)

  assert_allclose(vectors_rotated_in_new_frame, vectors_rotated_in_new_frame_expected)

@pytest.mark.parametrize('rot', rot_samples)
def test_calc_linear_displacement(rot, vector_samples):
  linear_displacement_scalars = rotex.calc_linear_displacement(rot, vector_samples)[1]

  rot_axis = rot.as_rotvec()
  rot_axis = rot_axis / np.linalg.norm(rot_axis)
  rot_angle = rot.magnitude()

  vectors_rotated= rot.apply(vector_samples)
  normals = np.cross(vector_samples, vectors_rotated)
  normals = normals / np.linalg.norm(normals, axis = 1)[:, np.newaxis]

  angles = np.arccos(np.dot(normals, rot_axis))
  vector_norms = np.linalg.norm(vector_samples, axis = 1)
  linear_displacement_scalars_expected = vector_norms * np.cos(angles) * rot_angle  # [ToDo] make sure it is right

  assert_allclose(linear_displacement_scalars, linear_displacement_scalars_expected, atol = 1e-2)

@pytest.mark.parametrize('rot', rot_in_small_angles_samples)
@pytest.mark.parametrize('delta_time', [1, 2, 3])
def test_calc_linear_velocity_in_small_angles(rot, vector_samples, delta_time):
  linear_velocities = rotex.calc_linear_velocity(rot, vector_samples, delta_time)[0]

  # calculate approx velocities by two vectors in small angles rotation
  vectors_rotated = rot.apply(vector_samples)
  linear_velocities_approx = (vectors_rotated - vector_samples) / delta_time

  assert_allclose(linear_velocities, linear_velocities_approx, atol = 1e-3)

@pytest.mark.parametrize('rot', rot_in_small_angles_samples)
@pytest.mark.parametrize('delta_time', [1, 2, 3])
def test_calc_centripetal_acceleration_in_small_angles(rot, vector_samples, delta_time):
  centripetal_acceleration_vectors = rotex.calc_centripetal_acceleration(rot, vector_samples, delta_time)[0]
  rot_axis = rot.as_rotvec()

  # angles between centripetal_acceleration and rot_axis should be all 90 degrees
  angles = np.arccos(np.dot(centripetal_acceleration_vectors, rot_axis) / (np.linalg.norm(rot_axis) * np.linalg.norm(centripetal_acceleration_vectors)))
  assert_allclose(angles, np.full(vector_samples.shape[0], np.deg2rad(90)), atol = 1e-8)

  # calculate approx acceleration by two vectors in small angles rotation
  vectors_rotated_before = rot.inv().apply(vector_samples)
  vectors_rotated_after = rot.apply(vector_samples)
  linear_velocities_before = (vector_samples - vectors_rotated_before) / delta_time
  linear_velocities_after = (vectors_rotated_after - vector_samples) / delta_time
  centripetal_acceleration_vectors_approx = (linear_velocities_after - linear_velocities_before) / delta_time

  assert_allclose(centripetal_acceleration_vectors, centripetal_acceleration_vectors_approx, atol = 1e-9)

#tests on other RotEx functions are covered in other test modules, like test_attitude.py
