import math
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

from helpers import attitude

from . import utils as test_utils
from . import test_rotate_vectors

def test_single_change_frame_ned_2_enu(frd_d_in_ned_frame, expected_frd_d_in_ned_frame):
  print('============================test single attitude change frame NED to ENU============================')

  (rot_in_enu_frame, rfu_d_in_enu_frame) = attitude.change_frame_ned_2_enu(frd_d_in_ned_frame, True)
  result = test_utils.get_result(np.allclose(rfu_d_in_enu_frame, expected_frd_d_in_ned_frame))
  print('***attitude from NED to ENU: %s***' % result)
  print('attitude FRD in NED frame: %s' % frd_d_in_ned_frame)
  print('attitude RFU in ENU fame: %s' % rfu_d_in_enu_frame)
  print('expected attitude RFU in ENU fame: %s\n' % expected_frd_d_in_ned_frame)

def test_single_from_enu_2_ned_frame(rfu_d_in_enu_frame, expected_rfu_d_in_enu_frame):
  print('============================test single attitude change frame ENU to NED============================')

  (rot_in_ned_frame, frd_d_in_ned_frame) = attitude.change_frame_enu_2_ned(rfu_d_in_enu_frame, True)
  result = test_utils.get_result(np.allclose(frd_d_in_ned_frame, expected_rfu_d_in_enu_frame))
  print('***attitude from ENU to NED: %s***' % result)
  print('attitude RFU in ENU frame: %s' % rfu_d_in_enu_frame)
  print('attitude FRD in NED fame: %s' % frd_d_in_ned_frame)
  print('expected attitude FRD in NED fame: %s\n' % expected_rfu_d_in_enu_frame)

def test_change_frame_ned_x_enu():
  test_single_change_frame_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  test_single_change_frame_ned_2_enu(np.array([0, 45, 0]), np.array([0, 45, 0]))
  test_single_change_frame_ned_2_enu(np.array([0, 0, 45]), np.array([0, 0, 45]))
  test_single_change_frame_ned_2_enu(np.array([90, 45, 90]), np.array([-90, 45, 90]))

  test_single_from_enu_2_ned_frame(np.array([-45, 0, 0]), np.array([45, 0, 0]))
  test_single_from_enu_2_ned_frame(np.array([0, 45, 0]), np.array([0, 45, 0]))
  test_single_from_enu_2_ned_frame(np.array([0, 0, 45]), np.array([0, 0, 45]))
  test_single_from_enu_2_ned_frame(np.array([-90, 45, 90]), np.array([90, 45, 90]))

def test_single_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle):
  print('============================test single attitude from heading in enu frame============================')

  print('heading as rfu: %s right_slope_angle: %s\n' % (heading_as_rfu, right_slope_angle))
  (rot, att_d_through_euler) = attitude.from_heading_in_enu_frame(heading_as_rfu, right_slope_angle, True)

  # test on heading vector by heading
  heading_start = np.array([0, 1, 0])
  heading_end_expected = heading_as_rfu / np.linalg.norm(heading_as_rfu)
  test_rotate_vectors.test_single_rotate_vectors_once(heading_start, att_d_through_euler, 'ZYX', heading_end_expected, False)

  # test on right slope angle by right direction
  right_start = np.array([1, 0, 0])
  right_end = rot.apply(right_start)
  right_slope_angle_result = math.atan2(right_end[2], math.sqrt(right_end[0] ** 2 + right_end[1] ** 2))
  right_slope_angle_result = np.rad2deg(right_slope_angle_result)

  result = test_utils.get_result(np.allclose(right_slope_angle, right_slope_angle_result))
  print('***right slope angle(deg): %s***' % result)
  print('input: %s' % right_slope_angle)
  print('result: %s\n' % right_slope_angle_result)

def test_from_heading_in_enu_frame():
  heading_as_rfu = np.array([-1, 1, math.sqrt(2)])
  right_slope_angle = 45
  test_single_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle)

  heading_as_rfu = np.array([-1, 2, 3])
  right_slope_angle = 0
  test_single_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle)

  heading_as_rfu = np.array([-1, 2, 0.5])
  right_slope_angle = 15
  test_single_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle)

def test_single_get_delta_att(att_d_1, att_d_2, rot_seq, in_world_frame):
  if in_world_frame:
    frame_str = 'world'
  else:
    frame_str = 'rot1'
  print('============================test single get delta attitude in %s frame============================' % frame_str)

  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rot2 = Rotation.from_euler(rot_seq, att_d_2, True)

  vectors = test_utils.get_test_vectors()

  if in_world_frame:
    vectors_by_rot1 = rot1.apply(vectors)
    vectors_by_rot2 = rot2.apply(vectors)
  else:
    vectors_by_rot1 = vectors # they are original coordinates in rot1 frame
    vectors_by_rot2 = rot2.apply(vectors)
    vectors_by_rot2 = rot1.inv().apply(vectors_by_rot2)

  (delta_rot, delta_euler_d) = attitude.get_delta_att(att_d_1, att_d_2, rot_seq, True, in_world_frame)
  vectors_by_delta_rot = delta_rot.apply(vectors_by_rot1)

  result = test_utils.get_result(np.allclose(vectors_by_rot2, vectors_by_delta_rot))
  print('***vectors rotated in %s frame: %s***' % (frame_str, result))
  print('by rot2: %s' % vectors_by_rot2)
  print('by delta_rot: %s\n' % vectors_by_delta_rot)

def test_single_linear_delta_att(att_d_1, factor, rot_seq):
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

  result = test_utils.get_result(np.allclose(delta_euler_d, delta_euler_d_linear))
  print('***delat att(deg) in %s sequence: %s***' % (rot_seq, result))
  print('attitude.delta_att(): %s' % delta_euler_d)
  print('linear_delta_att: %s\n' % delta_euler_d_linear)

def test_get_delta_att():
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])

  test_single_get_delta_att(att_d_1, att_d_2, 'ZYX', True)
  test_single_get_delta_att(att_d_1, att_d_2, 'ZYX', False)
  test_single_get_delta_att(att_d_1, att_d_2, 'zyx', True)
  test_single_get_delta_att(att_d_1, att_d_2, 'zyx', False)

  test_single_linear_delta_att(att_d_1, 1.1, 'ZYX')
  test_single_linear_delta_att(att_d_1, 1.2, 'zyx')

def test_single_calc_angular_velocity(att_d_1, att_d_2, rot_seq, delta_time):
  print('============================test single calc angular velocity============================')

  angular_velocity = attitude.calc_angular_velocity(att_d_1, att_d_2, rot_seq, True, delta_time, False)[0]

  times = [0, delta_time]
  angles = [att_d_1, att_d_2]
  rotations = Rotation.from_euler(rot_seq, angles, True)

  spline = RotationSpline(times, rotations)
  angular_velocity_spline = spline(times, 1)[1]
  angular_velocity_spline = np.rad2deg(angular_velocity_spline)

  result = test_utils.get_result(np.allclose(angular_velocity, angular_velocity_spline))
  print('***angular rate: %s***' % result)
  print('result: %s' % angular_velocity)
  print('spline: %s\n' % angular_velocity_spline)

def test_calc_angular_velocity():
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])
  test_single_calc_angular_velocity(att_d_1, att_d_2, 'ZYX', 2)
  test_single_calc_angular_velocity(att_d_1, att_d_2, 'zyx', 3)

def test():
  test_change_frame_ned_x_enu()
  test_from_heading_in_enu_frame()
  test_get_delta_att()
  test_calc_angular_velocity()
