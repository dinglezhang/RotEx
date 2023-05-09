import numpy as np

from helpers import common
from helpers import attitude as att

def test_one_att_ned_2_enu(att_d_ned_2_frd, expected_att_d_enu_2_rfu):
  att_d_enu_2_rfu = att.att_ned_2_enu(att_d_ned_2_frd, True)
  result = common.get_result(np.allclose(att_d_enu_2_rfu, expected_att_d_enu_2_rfu))
  print('***attitude from NED to ENU: %s***' % result)
  print('body attitude from NED to FRD:\n%s' % att_d_ned_2_frd)
  print('body attitude from ENU to RFU:\n%s\n' % att_d_enu_2_rfu)

def test_att_ned_2_enu():
  test_one_att_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  test_one_att_ned_2_enu(np.array([0, 45, 0]), np.array([0, 0, 45]))
  test_one_att_ned_2_enu(np.array([0, 0, 45]), np.array([0, 45, 0]))
  test_one_att_ned_2_enu(np.array([90, 45, 90]), np.array([0, 45, 90]))

def test_att_enu_2_rfu_by_delta_xyz():
  print('============================test att from enu to rfu by delta xyz and roll============================')

  delta_x = -1
  delta_y = 1
  delta_z = 0.01#math.sqrt(delta_x * delta_x + delta_y * delta_y)
  roll_y = 0.1
  print('delta_x: %s delta_y: %s delta_z: %s roll_y: %s\n' % (delta_x, delta_y, delta_z, roll_y))
  att.att_enu_2_rfu_by_euler(delta_x, delta_y, delta_z, roll_y)
  att.att_enu_2_rfu_by_vertical_rot_vec(delta_x, delta_y, delta_z, roll_y)

def test_delta_att():
  print('============================test delta att============================')

  att_d_input_1 = np.array([10, 1, 4])
  att_d_input_2 = np.array([11, 10, 5])

  att.delta_att(att_d_input_1, att_d_input_2, 'ZYX')
  att.delta_att(att_d_input_1, att_d_input_2, 'zyx')

def test():
  test_att_ned_2_enu()
  test_att_enu_2_rfu_by_delta_xyz()
  test_delta_att()
