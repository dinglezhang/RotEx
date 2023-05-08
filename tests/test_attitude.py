import numpy as np

from helpers import attitude as att

def test_att_ned_2_enu():
  att.att_ned_2_enu(np.array([45, 0, 0]), np.array([-45, 0, 0]))
  att.att_ned_2_enu(np.array([0, 45, 0]), np.array([0, 0, 45]))
  att.att_ned_2_enu(np.array([0, 0, 45]), np.array([0, 45, 0]))
  att.att_ned_2_enu(np.array([90, 45, 90]), np.array([0, 45, 90]))

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
