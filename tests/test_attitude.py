import math
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

from helpers import attitude

from . import test_util
from . import test_rotate_vectors

def test_single_att_ned_2_enu(att_d_ned_2_frd, expected_att_d_enu_2_rfu):
  print('============================test single attitude NED to ENU============================')

  att_d_enu_2_rfu = attitude.att_ned_2_enu(att_d_ned_2_frd, True)
  result = test_util.get_result(np.allclose(att_d_enu_2_rfu, expected_att_d_enu_2_rfu))
  print('***attitude from NED to ENU: %s***' % result)
  print('body attitude from NED to FRD:\n%s' % att_d_ned_2_frd)
  print('body attitude from ENU to RFU:\n%s\n' % att_d_enu_2_rfu)

def test_single_att_enu_2_ned(att_d_enu_2_rfu, expected_att_d_ned_2_frd):
  print('============================test single attitude ENU to NED============================')

  att_d_ned_2_frd = attitude.att_enu_2_ned(att_d_enu_2_rfu, True)
  result = test_util.get_result(np.allclose(att_d_ned_2_frd, expected_att_d_ned_2_frd))
  print('***attitude from ENU to NED: %s***' % result)
  print('body attitude from ENU to RFU:\n%s' % att_d_enu_2_rfu)
  print('body attitude from NED to FRD:\n%s\n' % att_d_ned_2_frd)

def test_att_ned_x_enu():
  test_single_att_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  test_single_att_ned_2_enu(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_single_att_ned_2_enu(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_single_att_ned_2_enu(np.array([90, 45, 90]), np.array([0, 45, 90]))

  test_single_att_enu_2_ned(np.array([-45, 0, 0]), np.array([45, 0, 0]))
  test_single_att_enu_2_ned(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_single_att_enu_2_ned(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_single_att_enu_2_ned(np.array([0, 45, 90]), np.array([90, 45, 90]))

def test_att_enu_2_rfu_by_delta_xyz():
  print('============================test attitude from enu to rfu by delta xyz and cross slope angle============================')

  delta_x = -1
  delta_y = 2
  delta_z = 0.5#math.sqrt(delta_x * delta_x + delta_y * delta_y)
  cross_slope_angle = 15
  print('delta_x: %s delta_y: %s delta_z: %s cross_slope_angle: %s\n' % (delta_x, delta_y, delta_z, cross_slope_angle))

  (rot, att_d_through_euler) = attitude.att_enu_2_rfu(delta_x, delta_y, delta_z, cross_slope_angle, True)

  # test by heading frond
  heading_front_start = np.array([0, 1, 0])
  heading_front_end_expected = np.array([delta_x, delta_y, delta_z])
  heading_front_end_expected = heading_front_end_expected / np.linalg.norm(heading_front_end_expected)
  test_rotate_vectors.test_single_rotate_vectors_once(heading_front_start, att_d_through_euler, 'ZYX', heading_front_end_expected, False)

  # test cross slope angle by heading right
  heading_right_start = np.array([1, 0, 0])
  heading_right_end = rot.apply(heading_right_start)
  cross_slope_angle_calc = math.atan2(heading_right_end[2], math.sqrt(heading_right_end[0] ** 2 + heading_right_end[1] ** 2))
  cross_slope_angle_calc = np.rad2deg(cross_slope_angle_calc)

  result = test_util.get_result(np.allclose(cross_slope_angle, cross_slope_angle_calc))
  print('***cross slope angle(deg): %s***' % result)
  print('input: %s' % cross_slope_angle)
  print('calculated: %s\n' % cross_slope_angle_calc)

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
  test_att_ned_x_enu()
  test_att_enu_2_rfu_by_delta_xyz()
  test_delta_att()
  test_angular_rate()
