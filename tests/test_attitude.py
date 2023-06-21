import math
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, RotationSpline

from EasyEuler import attitude

from . import test_rotate_vectors

def single_test_change_frame_ned_2_enu(frd_d_in_ned_frame, expected_frd_d_in_ned_frame):
  print('============================test single attitude change frame NED to ENU============================')

  (rot_in_enu_frame, rfu_d_in_enu_frame) = attitude.change_frame_ned_2_enu(frd_d_in_ned_frame, True)
  assert_allclose(rfu_d_in_enu_frame, expected_frd_d_in_ned_frame)

def single_test_from_enu_2_ned_frame(rfu_d_in_enu_frame, expected_rfu_d_in_enu_frame):
  print('============================test single attitude change frame ENU to NED============================')

  (rot_in_ned_frame, frd_d_in_ned_frame) = attitude.change_frame_enu_2_ned(rfu_d_in_enu_frame, True)
  assert_allclose(frd_d_in_ned_frame, expected_rfu_d_in_enu_frame, atol=1e-8)

def test_change_frame_ned_x_enu():
  single_test_change_frame_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  single_test_change_frame_ned_2_enu(np.array([0, 45, 0]), np.array([0, 45, 0]))
  single_test_change_frame_ned_2_enu(np.array([0, 0, 45]), np.array([0, 0, 45]))
  single_test_change_frame_ned_2_enu(np.array([90, 45, 90]), np.array([-90, 45, 90]))

  single_test_from_enu_2_ned_frame(np.array([-45, 0, 0]), np.array([45, 0, 0]))
  single_test_from_enu_2_ned_frame(np.array([0, 45, 0]), np.array([0, 45, 0]))
  single_test_from_enu_2_ned_frame(np.array([0, 0, 45]), np.array([0, 0, 45]))
  single_test_from_enu_2_ned_frame(np.array([-90, 45, 90]), np.array([90, 45, 90]))

def single_test_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle):
  print('============================test single attitude from heading in enu frame============================')

  print('heading as rfu: %s right_slope_angle: %s\n' % (heading_as_rfu, right_slope_angle))
  (rot, att_d_through_euler) = attitude.from_heading_in_enu_frame(heading_as_rfu, right_slope_angle, True)

  # test on heading vector by heading
  heading_start = np.array([0, 1, 0])
  heading_end_expected = heading_as_rfu / np.linalg.norm(heading_as_rfu)
  test_rotate_vectors.single_test_rotate_vectors_once(heading_start, att_d_through_euler, 'ZYX', heading_end_expected, False)

  # test on right slope angle by right direction
  right_start = np.array([1, 0, 0])
  right_end = rot.apply(right_start)
  right_slope_angle_result = math.atan2(right_end[2], math.sqrt(right_end[0] ** 2 + right_end[1] ** 2))
  right_slope_angle_result = np.rad2deg(right_slope_angle_result)

  assert_allclose(right_slope_angle_result, right_slope_angle)

def test_from_heading_in_enu_frame():
  heading_as_rfu = np.array([-1, 1, math.sqrt(2)])
  right_slope_angle = 45
  single_test_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle)

  heading_as_rfu = np.array([-1, 2, 3])
  right_slope_angle = 0
  single_test_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle)

  heading_as_rfu = np.array([-1, 2, 0.5])
  right_slope_angle = 15
  single_test_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle)

def single_test_get_delta_att(att_d_1, att_d_2, rot_seq, in_world_frame, vector_samples):
  if in_world_frame:
    frame_str = 'world'
  else:
    frame_str = 'rot1'
  print('============================test single get delta attitude in %s frame============================' % frame_str)

  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rot2 = Rotation.from_euler(rot_seq, att_d_2, True)

  if in_world_frame:
    vectors_by_rot1 = rot1.apply(vector_samples)
    vectors_by_rot2 = rot2.apply(vector_samples)
  else:
    vectors_by_rot1 = vector_samples # they are original coordinates in rot1 frame
    vectors_by_rot2 = rot2.apply(vector_samples)
    vectors_by_rot2 = rot1.inv().apply(vectors_by_rot2)

  (delta_rot, delta_euler_d) = attitude.get_delta_att(att_d_1, att_d_2, rot_seq, True, in_world_frame)
  vectors_by_delta_rot = delta_rot.apply(vectors_by_rot1)

  assert_allclose(vectors_by_rot2, vectors_by_delta_rot)

def single_test_linear_delta_att(att_d_1, factor, rot_seq):
  print('============================test single linear delta attitude============================')

  # calculate att_d_2 from att_d_1 and factor (linear change on rotvec)
  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rotvec1 = rot1.as_rotvec()
  rotvec2 = rotvec1 * factor
  rot2 = Rotation.from_rotvec(rotvec2)
  att_d_2 = rot2.as_euler(rot_seq, True)

  # calculate delta_att by attitude.delta_att()
  (delta_rot, delta_euler_d) = attitude.get_delta_att(att_d_1, att_d_2, rot_seq, True, False)

  # calcaulate delta_att_linear by a special way for linear rotvec change
  delta_rotvec_linear = rotvec2 - rotvec1
  delta_rot_linear = Rotation.from_rotvec(delta_rotvec_linear)
  delta_euler_d_linear = delta_rot_linear.as_euler(rot_seq, True)

  assert_allclose(delta_euler_d, delta_euler_d_linear)

def test_get_delta_att(vector_samples):
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])

  single_test_get_delta_att(att_d_1, att_d_2, 'ZYX', True, vector_samples)
  single_test_get_delta_att(att_d_1, att_d_2, 'ZYX', False, vector_samples)
  single_test_get_delta_att(att_d_1, att_d_2, 'zyx', True, vector_samples)
  single_test_get_delta_att(att_d_1, att_d_2, 'zyx', False, vector_samples)

  single_test_linear_delta_att(att_d_1, 1.1, 'ZYX')
  single_test_linear_delta_att(att_d_1, 1.2, 'zyx')

def single_test_calc_angular_velocity(att_d_1, att_d_2, rot_seq, delta_time):
  print('============================test single calc angular velocity============================')

  angular_velocity = attitude.calc_angular_velocity(att_d_1, att_d_2, rot_seq, True, delta_time, False)[0]

  times = [0, delta_time]
  angles = [att_d_1, att_d_2]
  rotations = Rotation.from_euler(rot_seq, angles, True)

  spline = RotationSpline(times, rotations)
  angular_velocity_spline = spline(times, 1)[1]
  angular_velocity_spline = np.rad2deg(angular_velocity_spline)

  assert_allclose(angular_velocity, angular_velocity_spline)

def test_calc_angular_velocity():
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])
  single_test_calc_angular_velocity(att_d_1, att_d_2, 'ZYX', 2)
  single_test_calc_angular_velocity(att_d_1, att_d_2, 'zyx', 3)
