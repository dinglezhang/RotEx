import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import common
from . import new_frame

logger = common.get_logger()

'''
  NED_2_ENU and ENU_2_NED are actually the same rotation.
  So define NED_X_ENU which means to exchange each other
'''
EULER_D_FRAME_NED_X_ENU = np.array([-90, 180, 0])
EULER_R_FRAME_NED_X_ENU = EULER_D_FRAME_NED_X_ENU * common.D2R

def att_ned_x_enu(att_old_frame, is_degree):
  euler_frame_ned_x_enu = EULER_D_FRAME_NED_X_ENU
  if not is_degree:
    euler_frame_ned_x_enu = EULER_R_FRAME_NED_X_ENU

  att_new_frame = new_frame.euler_in_new_frame(att_old_frame, euler_frame_ned_x_enu, 'ZYX', is_degree)

  return att_new_frame

'''
Get attitude in ENU (East, North, Up) frame from NED (North, East, Down) frame.

Args:
  att_ned_2_frd: body attitude FRD (Front, Right, Down) in NED frame
  is_degree: True is degree, False is radian
Return:
  body attitude RFU (Right, Front, Up) in ENU frame
'''
def att_ned_2_enu(att_ned_2_frd, is_degree):
  unit = 'deg'
  if not is_degree:
    unit = 'rad'

  logger.info('body attitude FRD in NED frame: euler(%s)%s' % (unit, att_ned_2_frd))
  att_enu_2_rfu = att_ned_x_enu(att_ned_2_frd, is_degree)
  logger.info('body attitude RFU in ENU frame: euler(%s)%s' % (unit, att_enu_2_rfu))

  return att_enu_2_rfu

'''
Get attitude in NED (North, East, Down) frame from ENU (East, North, Up) frame.

Args:
  att_enu_2_rfu: body attitude RFU (Right, Front, Up) in NED frame
  is_degree: True is degree, False is radian
Return:
  body attitude FRD (Front, Right, Down) in NED frame
'''
def att_enu_2_ned(att_enu_2_rfu, is_degree):
  unit = 'deg'
  if not is_degree:
    unit = 'rad'

  logger.info('body attitude RFU in ENU frame: euler(%s)%s' % (unit, att_enu_2_rfu))
  att_ned_2_frd = att_ned_x_enu(att_enu_2_rfu, is_degree)
  logger.info('body attitude FRD in NED frame: euler(%s)%s' % (unit, att_ned_2_frd))

  return att_ned_2_frd

'''
  rotation from enu (navigation frame) to rfu (body frame), enu / rfu is xyz of coordinates
  heading vector starts from direction along with north and ends to direction along with its front orient
  delta_x, delta_y, delta_z indicate tangent direction, which is end direction of heading
  roll_y indicates self rotation of the heading vector
'''
'''
  way 1: by euler
'''
def att_enu_2_rfu_by_euler(delta_x, delta_y, delta_z, roll_y):
  print('~~~~~~~~~~att from enu to rfu by delta xyz and roll, way 1: euler~~~~~~~~~~')

  yaw_z = math.atan2(-delta_x, delta_y)
  pitch_x = math.atan2(delta_z, math.sqrt(delta_x * delta_x + delta_y * delta_y))

  euler_r_delta_xyz = np.array([yaw_z, pitch_x, roll_y])
  euler_d_delta_xyz = euler_r_delta_xyz * common.R2D
  print('euler calculated by delta xyz by zxy sequence intrinsicly:')
  print('%s (%s)' % (euler_d_delta_xyz, euler_r_delta_xyz))

  rot = Rotation.from_euler("ZXY", euler_r_delta_xyz)

  euler_r_rotated = rot.as_euler('ZYX')
  euler_d_rotated = euler_r_rotated * common.R2D
  print('euler rotated by zyx sequence:')
  print('%s (%s)\n' % (euler_d_rotated, euler_r_rotated))

  # test by heading vector
  heading_start = np.array([0, 1, 0])
  heading_end = np.array([delta_x, delta_y, delta_z])
  heading_end = heading_end / np.linalg.norm(heading_end)

  heading_end_rotated = rot.apply(heading_start)
  result = common.get_result(np.allclose(heading_end, heading_end_rotated))
  print('***heading rotated by euler: %s***' % result)
  print('end heading expected: %s' % heading_end)
  print('end heading rotated: %s\n' % heading_end_rotated)

def get_vertical_rot_vec(v1, v2):
  rot_vec = np.cross(v1, v2)
  rot_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
  rot_vec = rot_vec / np.linalg.norm(rot_vec) * rot_angle
  return rot_vec

'''
  way 2: by vertical rot vec
  [ToDo] how to use roll_y
'''
def att_enu_2_rfu_by_vertical_rot_vec(delta_x, delta_y, delta_z, roll_y):
  print('~~~~~~~~~~att from enu to rfu by delta xyz and roll, way 2: by vertical rot vec~~~~~~~~~~')

  heading_start = np.array([0, 1, 0])
  heading_end = np.array([delta_x, delta_y, delta_z])
  heading_end = heading_end / np.linalg.norm(heading_end)

  vertical_rot_vec = get_vertical_rot_vec(heading_start, heading_end)

  rot = Rotation.from_rotvec(vertical_rot_vec)
  euler_r_roated = rot.as_euler('ZYX')
  euler_d_rotated = euler_r_roated * common.R2D
  print('euler rotated by zyx sequence:')
  print('%s (%s)\n' % (euler_d_rotated, euler_r_roated))

  heading_end_rotated = rot.apply(heading_start)
  result = common.get_result(np.allclose(heading_end, heading_end_rotated))
  print('***heading rotated by vertical vec: %s***' % result)
  print('end heading expected: %s' % heading_end)
  print('end heading rotated: %s\n' % heading_end_rotated)

def delta_att(att_d_input_1, att_d_input_2, rot_seq):
  print('~~~~~~~~~~delta att~~~~~~~~~~')

  att_r_input_1 = att_d_input_1 * common.D2R
  att_r_input_2 = att_d_input_2 * common.D2R
  print('att input by %s sequence:' % rot_seq)
  print('%s (%s)' % (att_d_input_1, att_r_input_1))
  print('%s (%s)\n' % (att_d_input_2, att_r_input_2))

  rot1 = Rotation.from_euler(rot_seq, att_r_input_1)
  rot2 = Rotation.from_euler(rot_seq, att_r_input_2)

  delta_rot = rot1.inv() * rot2

  delta_att_d = delta_rot.as_euler(rot_seq, True)
  delta_att_r = delta_rot.as_euler(rot_seq)
  print('delta att by %s sequence :' % rot_seq)
  print('%s (%s)' % (delta_att_d, delta_att_r))

  #euler_in_new_frame(delta_rot.as_euler(rot_seq, True), att_d_input_1, rot_seq)

  # test
  rot2_cal = rot1 * delta_rot
  att_d_2_cal = rot2_cal.as_euler(rot_seq, True)
  att_r_2_cal = rot2_cal.as_euler(rot_seq)
  print('att2 calculated by %s sequence :' % rot_seq)
  print('%s (%s)' % (att_d_2_cal, att_r_2_cal))
  result = common.get_result(np.allclose(att_d_2_cal, att_d_input_2))
  print('***att_d_input_2 vs att_d_cal_2: %s***\n' % result)
