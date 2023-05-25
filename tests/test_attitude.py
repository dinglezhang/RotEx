import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

from helpers import attitude as att
from . import test_util

def test_one_att_ned_2_enu(att_d_ned_2_frd, expected_att_d_enu_2_rfu):
  print('============================test one attitude NED to ENU============================')

  att_d_enu_2_rfu = att.att_ned_2_enu(att_d_ned_2_frd, True)
  result = test_util.get_result(np.allclose(att_d_enu_2_rfu, expected_att_d_enu_2_rfu))
  print('***attitude from NED to ENU: %s***' % result)
  print('body attitude from NED to FRD:\n%s' % att_d_ned_2_frd)
  print('body attitude from ENU to RFU:\n%s\n' % att_d_enu_2_rfu)

def test_one_att_enu_2_ned(att_d_enu_2_rfu, expected_att_d_ned_2_frd):
  print('============================test one attitude ENU to NED============================')

  att_d_ned_2_frd = att.att_enu_2_ned(att_d_enu_2_rfu, True)
  result = test_util.get_result(np.allclose(att_d_ned_2_frd, expected_att_d_ned_2_frd))
  print('***attitude from ENU to NED: %s***' % result)
  print('body attitude from ENU to RFU:\n%s' % att_d_enu_2_rfu)
  print('body attitude from NED to FRD:\n%s\n' % att_d_ned_2_frd)

def test_att_ned_x_enu():
  test_one_att_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  test_one_att_ned_2_enu(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_one_att_ned_2_enu(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_one_att_ned_2_enu(np.array([90, 45, 90]), np.array([0, 45, 90]))

  test_one_att_enu_2_ned(np.array([-45, 0, 0]), np.array([45, 0, 0]))
  test_one_att_enu_2_ned(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_one_att_enu_2_ned(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_one_att_enu_2_ned(np.array([0, 45, 90]), np.array([90, 45, 90]))

def test_att_enu_2_rfu_by_delta_xyz():
  print('============================test attitude from enu to rfu by delta xyz and roll============================')

  delta_x = -1
  delta_y = 1
  delta_z = 0.01#math.sqrt(delta_x * delta_x + delta_y * delta_y)
  roll_y = 0.1
  print('delta_x: %s delta_y: %s delta_z: %s roll_y: %s\n' % (delta_x, delta_y, delta_z, roll_y))

  # test by heading vector
  heading_north = np.array([0, 1, 0])
  heading_expected = np.array([delta_x, delta_y, delta_z])
  heading_expected = heading_expected / np.linalg.norm(heading_expected)

  att_d_through_euler = att.att_enu_2_rfu_through_euler(delta_x, delta_y, delta_z, roll_y, True)
  rot_through_euler = Rotation.from_euler("ZYX", att_d_through_euler, True)
  heading_rotated_through_euler = rot_through_euler.apply(heading_north)
  result = test_util.get_result(np.allclose(heading_expected, heading_rotated_through_euler))
  print('***heading through euler: %s***' % result)
  print('expected: %s' % heading_expected)
  print('rotated: %s\n' % heading_rotated_through_euler)

  att_d_through_rotvec = att.att_enu_2_rfu_through_rotvec(delta_x, delta_y, delta_z, roll_y, True)
  rot_through_rotvec = Rotation.from_euler("ZYX", att_d_through_rotvec, True)
  heading_rotated_through_rotvec = rot_through_rotvec.apply(heading_north)
  result = test_util.get_result(np.allclose(heading_expected, heading_rotated_through_rotvec))
  print('***heading through vertical rotvec: %s***' % result)
  print('expected: %s' % heading_expected)
  print('rotated: %s\n' % heading_rotated_through_rotvec)

  #[ToDo] add more test on vectors other than heading_north

def test_one_delta_att(att_d_1, att_d_2, rot_seq):
  print('============================test one delta attitude============================')

  (delta_rot, delta_euler_d) = att.delta_att(att_d_1, att_d_2, rot_seq, True)

  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rot2_calc = rot1 * delta_rot

  att_d_2_calc = rot2_calc.as_euler(rot_seq, True)
  result = test_util.get_result(np.allclose(att_d_2_calc, att_d_2))
  print('***att2(deg) by %s sequence after rotation by delta euler: %s***' % (rot_seq, result))
  print('expected: %s' % att_d_2)
  print('rotated: %s\n' % att_d_2_calc)

def test_delta_att():
  att_d_1 = np.array([10, 1, 4])
  att_d_2 = np.array([11, 2, 5])
  test_one_delta_att(att_d_1, att_d_2, 'ZYX')
  test_one_delta_att(att_d_1, att_d_2, 'zyx')

def test_one_angular_rate(delta_time, att_d_1, att_d_2, rot_seq):
  print('============================test one angular rate============================')

  angular_rate = att.angular_rate(delta_time, att_d_1, att_d_2, rot_seq, True)

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
  test_one_angular_rate(2, att_d_1, att_d_2, 'ZYX')
  test_one_angular_rate(3, att_d_1, att_d_2, 'zyx')

def test():
  test_att_ned_x_enu()
  test_att_enu_2_rfu_by_delta_xyz()
  test_delta_att()
  test_angular_rate()
