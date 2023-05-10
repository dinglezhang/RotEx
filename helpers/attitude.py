import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import common
from . import new_frame

logger = common.get_logger()

'''
  Attitude is a set of euler angles how a body intrinsic rotates from the navigation frame to its current body frame.

  Navigation frame (xyz coordinates):
    ENU (East, North, Up)`
    NED (North, East, Down)

  Body frame (xyz coordinates):
    RFU (Right, Front, Up)
    FRD (Front, Right, Down)
'''

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
Get attitude in ENU frame from NED frame.

Args:
  att_ned_2_frd: body attitude FRD in NED frame
  is_degree: True is degree, False is radian
Return:
  body attitude RFU in ENU frame
'''
def att_ned_2_enu(att_ned_2_frd, is_degree):
  logger.info('attitude FRD in NED frame: euler(%s)%s' % (common.get_angle_unit(is_degree), att_ned_2_frd))
  att_enu_2_rfu = att_ned_x_enu(att_ned_2_frd, is_degree)
  logger.info('attitude RFU in ENU frame: euler(%s)%s' % (common.get_angle_unit(is_degree), att_enu_2_rfu))

  return att_enu_2_rfu

'''
Get attitude in NED frame from ENU frame.

Args:
  att_enu_2_rfu: body attitude RFU in NED frame
  is_degree: True is degree, False is radian
Return:
  body attitude FRD in NED frame
'''
def att_enu_2_ned(att_enu_2_rfu, is_degree):
  logger.info('attitude RFU in ENU frame: euler(%s)%s' % (common.get_angle_unit(is_degree), att_enu_2_rfu))
  att_ned_2_frd = att_ned_x_enu(att_enu_2_rfu, is_degree)
  logger.info('attitude FRD in NED frame: euler(%s)%s' % (common.get_angle_unit(is_degree), att_ned_2_frd))

  return att_ned_2_frd

'''
Get attitude by delta x/y/z and roll on y in ENU frame.
Say a vector like body heading vector, it rotates from north direction to its front direction
The function is through the way by calculation of euler by delta x/y/z

Args:
  delta_x, delta_y, delta_z: tangent direction, which is end direction of body heading vector
  roll_y: self rotation of the heading vector
  is_degree: result returned is degree if True, radian if False
Return:
  body attitude RFU in ENU frame
'''
def att_enu_2_rfu_through_euler(delta_x, delta_y, delta_z, roll_y, is_degree):
  yaw_z = math.atan2(-delta_x, delta_y)
  pitch_x = math.atan2(delta_z, math.sqrt(delta_x * delta_x + delta_y * delta_y))

  att_r_ZXY = np.array([yaw_z, pitch_x, roll_y])
  logger.info('attitude by ZXY sequence: euler(rad)%s' % att_r_ZXY)

  rot = Rotation.from_euler("ZXY", att_r_ZXY, False)

  att_ZYX = rot.as_euler('ZYX', is_degree)
  logger.info('attitude by ZYX sequence: euler(%s)%s' % (common.get_angle_unit(is_degree), att_ZYX))

  return att_ZYX

def get_vertical_rotvec(v1, v2):
  rot_vec = np.cross(v1, v2)
  rot_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
  rot_vec = rot_vec / np.linalg.norm(rot_vec) * rot_angle
  return rot_vec

'''
Get attitude by delta x/y/z and roll on y in ENU frame.
Say a vector like body heading vector, it rotates from north direction to its front direction
The function is through the way by calculation of vertical rotvec by delta x/y/z
[ToDo] how to use roll_y

Args:
  delta_x, delta_y, delta_z: tangent direction, which is end direction of body heading vector
  roll_y: self rotation of the heading vector
  is_degree: result returned is degree if True, radian if False
Return:
  body attitude RFU in ENU frame
'''
def att_enu_2_rfu_through_rotvec(delta_x, delta_y, delta_z, roll_y, is_degree):
  heading_start = np.array([0, 1, 0])
  heading_end = np.array([delta_x, delta_y, delta_z])
  heading_end = heading_end / np.linalg.norm(heading_end)

  vertical_rotvec = get_vertical_rotvec(heading_start, heading_end)

  rot = Rotation.from_rotvec(vertical_rotvec)
  att = rot.as_euler('ZYX', is_degree)
  logger.info('attitude by ZYX sequence: euler(%s)%s' % (common.get_angle_unit(is_degree), att))

  return att

def delta_att(att_d_input_1, att_d_input_2, rot_seq):
  logger.info('~~~~~~~~~~delta att~~~~~~~~~~')

  att_r_input_1 = att_d_input_1 * common.D2R
  att_r_input_2 = att_d_input_2 * common.D2R
  logger.info('att input by %s sequence:' % rot_seq)
  logger.info('%s (%s)' % (att_d_input_1, att_r_input_1))
  logger.info('%s (%s)\n' % (att_d_input_2, att_r_input_2))

  rot1 = Rotation.from_euler(rot_seq, att_r_input_1)
  rot2 = Rotation.from_euler(rot_seq, att_r_input_2)

  delta_rot = rot1.inv() * rot2

  delta_att_d = delta_rot.as_euler(rot_seq, True)
  delta_att_r = delta_rot.as_euler(rot_seq)
  logger.info('delta att by %s sequence :' % rot_seq)
  logger.info('%s (%s)' % (delta_att_d, delta_att_r))

  #euler_in_new_frame(delta_rot.as_euler(rot_seq, True), att_d_input_1, rot_seq)

  # test
  rot2_cal = rot1 * delta_rot
  att_d_2_cal = rot2_cal.as_euler(rot_seq, True)
  att_r_2_cal = rot2_cal.as_euler(rot_seq)
  logger.info('att2 calculated by %s sequence :' % rot_seq)
  logger.info('%s (%s)' % (att_d_2_cal, att_r_2_cal))
  result = common.get_result(np.allclose(att_d_2_cal, att_d_input_2))
  logger.info('***att_d_input_2 vs att_d_cal_2: %s***\n' % result)
