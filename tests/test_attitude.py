import math
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

from helpers import attitude

from . import test_util
from . import test_rotate_vectors

def test_single_from_ned_2_enu_frame(att_d_ned_2_frd, expected_att_d_enu_2_rfu):
  print('============================test single attitude from NED to ENU frame============================')

  att_d_enu_2_rfu = attitude.from_ned_2_enu_frame(att_d_ned_2_frd, True)
  result = test_util.get_result(np.allclose(att_d_enu_2_rfu, expected_att_d_enu_2_rfu))
  print('***attitude from NED to ENU: %s***' % result)
  print('body attitude from NED to FRD:\n%s' % att_d_ned_2_frd)
  print('body attitude from ENU to RFU:\n%s\n' % att_d_enu_2_rfu)

def test_single_from_enu_2_ned_frame(att_d_enu_2_rfu, expected_att_d_ned_2_frd):
  print('============================test single attitude from ENU to NED frame============================')

  att_d_ned_2_frd = attitude.from_enu_2_ned_frame(att_d_enu_2_rfu, True)
  result = test_util.get_result(np.allclose(att_d_ned_2_frd, expected_att_d_ned_2_frd))
  print('***attitude from ENU to NED: %s***' % result)
  print('body attitude from ENU to RFU:\n%s' % att_d_enu_2_rfu)
  print('body attitude from NED to FRD:\n%s\n' % att_d_ned_2_frd)

def test_from_ned_x_enu_frame():
  test_single_from_ned_2_enu_frame(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  test_single_from_ned_2_enu_frame(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_single_from_ned_2_enu_frame(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_single_from_ned_2_enu_frame(np.array([90, 45, 90]), np.array([0, 45, 90]))

  test_single_from_enu_2_ned_frame(np.array([-45, 0, 0]), np.array([45, 0, 0]))
  test_single_from_enu_2_ned_frame(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_single_from_enu_2_ned_frame(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_single_from_enu_2_ned_frame(np.array([0, 45, 90]), np.array([90, 45, 90]))

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

  result = test_util.get_result(np.allclose(right_slope_angle, right_slope_angle_result))
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

def test_single_delta_att(att_d_1, att_d_2, rot_seq):
  print('============================test single delta attitude============================')

  (delta_rot, delta_euler_d) = attitude.delta_att(att_d_1, att_d_2, rot_seq, True)

  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rot2_calc = rot1 * delta_rot

  att_d_2_calc = rot2_calc.as_euler(rot_seq, True)
  result = test_util.get_result(np.allclose(att_d_2_calc, att_d_2))
  print('***att2(deg) in %s sequence after rotation by delta euler: %s***' % (rot_seq, result))
  print('expected: %s' % att_d_2)
  print('rotated: %s\n' % att_d_2_calc)

def test_linear_delta_att(att_d_1, factor, rot_seq):
  print('============================test linear delta attitude============================')

  # calculate att_d_2 from att_d_1 and factor (linear change on rotvec)
  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rotvec1 = rot1.as_rotvec()
  rotvec2 = rotvec1 * factor
  rot2 = Rotation.from_rotvec(rotvec2)
  att_d_2 = rot2.as_euler(rot_seq, True)

  # calculate delta_att by attitude.delta_att()
  (delta_rot, delta_euler_d) = attitude.delta_att(att_d_1, att_d_2, rot_seq, True)

  # calcaulate delta_att_linear by a special way for linear rotvec change
  delta_rotvec_linear = rotvec2 - rotvec1
  delta_rot_linear = Rotation.from_rotvec(delta_rotvec_linear)
  delta_euler_d_linear = delta_rot_linear.as_euler(rot_seq, True)

  result = test_util.get_result(np.allclose(delta_euler_d, delta_euler_d_linear))
  print('***delat att(deg) in %s sequence: %s***' % (rot_seq, result))
  print('attitude.delta_att(): %s' % delta_euler_d)
  print('linear_delta_att: %s\n' % delta_euler_d_linear)

def test_delta_att():
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])

  test_single_delta_att(att_d_1, att_d_2, 'ZYX')
  test_single_delta_att(att_d_1, att_d_2, 'zyx')

  test_linear_delta_att(att_d_1, 1.1, 'ZYX')
  test_linear_delta_att(att_d_1, 1.2, 'zyx')

def test_single_angular_rate(delta_time, att_d_1, att_d_2, rot_seq):
  print('============================test single angular rate============================')

  angular_rate = attitude.angular_rate(delta_time, att_d_1, att_d_2, rot_seq, True)

  times = [0, delta_time]
  angles = [att_d_1, att_d_2]
  rotations = Rotation.from_euler(rot_seq, angles, True)

  spline = RotationSpline(times, rotations)
  angular_rate_spline = spline(times, 1)[1]
  angular_rate_spline = np.rad2deg(angular_rate_spline)

  result = test_util.get_result(np.allclose(angular_rate_spline, angular_rate))
  print('***angular rate: %s***' % result)
  print('spline: %s' % angular_rate_spline)
  print('actual: %s\n' % angular_rate)

def test_angular_rate():
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])
  test_single_angular_rate(2, att_d_1, att_d_2, 'ZYX')
  test_single_angular_rate(3, att_d_1, att_d_2, 'zyx')

def test():
  test_from_ned_x_enu_frame()
  test_from_heading_in_enu_frame()
  test_delta_att()
  test_angular_rate()
