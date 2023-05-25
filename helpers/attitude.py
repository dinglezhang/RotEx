import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import util
from . import new_frame

logger = util.get_logger()

'''
  Attitude is a set of euler angles how a body intrinsic rotates from the world frame to its current body frame.

  World frame (xyz coordinates):
    ENU (East, North, Up)`
    NED (North, East, Down)

  Body frame (xyz coordinates):
    RFU (Right, Front, Up)
    FRD (Front, Right, Down)
'''

'''
  NED_2_ENU and ENU_2_NED are actually the same rotation.
  So define NED_X_ENU which means to exchange each other, which rotation sequence is 'ZYX'
'''
EULER_D_FRAME_NED_X_ENU_ZYX = np.array([-90, 180, 0])
EULER_R_FRAME_NED_X_ENU_ZYX = np.deg2rad(EULER_D_FRAME_NED_X_ENU_ZYX)

def att_ned_x_enu(att_old_frame, is_degree):
  euler_frame_ned_x_enu = EULER_D_FRAME_NED_X_ENU_ZYX
  if not is_degree:
    euler_frame_ned_x_enu = EULER_R_FRAME_NED_X_ENU_ZYX

  att_new_frame = new_frame.euler_in_new_frame(att_old_frame, euler_frame_ned_x_enu, 'ZYX', is_degree)

  return att_new_frame

'''
Get attitude in ENU frame from NED frame.

Args:
  att_ned_2_frd: body attitude FRD in NED frame
  is_degree: True is degree and False is radian for both input and output attitudes
Return:
  body attitude RFU in ENU frame
'''
def att_ned_2_enu(att_ned_2_frd, is_degree):
  logger.info('attitude FRD in NED frame: euler(%s)%s' % (util.get_angle_unit(is_degree), att_ned_2_frd))
  att_enu_2_rfu = att_ned_x_enu(att_ned_2_frd, is_degree)
  logger.info('attitude RFU in ENU frame: euler(%s)%s' % (util.get_angle_unit(is_degree), att_enu_2_rfu))

  return att_enu_2_rfu

'''
Get attitude in NED frame from ENU frame.

Args:
  att_enu_2_rfu: body attitude RFU in NED frame
  is_degree: True is degree and False is radian for both input and output attitudes
Return:
  body attitude FRD in NED frame
'''
def att_enu_2_ned(att_enu_2_rfu, is_degree):
  logger.info('attitude RFU in ENU frame: euler(%s)%s' % (util.get_angle_unit(is_degree), att_enu_2_rfu))
  att_ned_2_frd = att_ned_x_enu(att_enu_2_rfu, is_degree)
  logger.info('attitude FRD in NED frame: euler(%s)%s' % (util.get_angle_unit(is_degree), att_ned_2_frd))

  return att_ned_2_frd

'''
Get attitude by delta x/y/z and roll on y in ENU frame.
Say a vector like body heading vector, it rotates from north direction to its front direction
The function is through the way by calculation of euler by delta x/y/z

Args:
  delta_x, delta_y, delta_z: tangent direction, which is end direction of body heading vector
  roll_y: self rotation of the heading vector
  is_degree: True is degree and False is radian for output attitude
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
  logger.info('attitude by ZYX sequence: euler(%s)%s' % (util.get_angle_unit(is_degree), att_ZYX))

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
  is_degree: True is degree and False is radian for output attitude
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
  logger.info('attitude by ZYX sequence: euler(%s)%s' % (util.get_angle_unit(is_degree), att))

  return att

'''
Get delta between two attitudes.

Args:
  att1, att2: two attitudes
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for input attitudes and output delta euler
Return:
  [0]: delt rotation between two attitudes
  [1]: delta euler between two attitudes
'''
def delta_att(att1, att2, rot_seq, is_degree):
  angle_unit = util.get_angle_unit(is_degree)
  logger.info('two attitudes input by %s sequence:' % rot_seq)
  logger.info('euler1(%s)%s' % (angle_unit, att1))
  logger.info('euler2(%s)%s' % (angle_unit, att2))

  rot1 = Rotation.from_euler(rot_seq, att1, is_degree)
  rot2 = Rotation.from_euler(rot_seq, att2, is_degree)

  delta_rot = rot1.inv() * rot2

  delta_euler = delta_rot.as_euler(rot_seq, is_degree)
  logger.info('delta euler by %s sequence: euler(%s)%s' % (rot_seq, angle_unit, delta_euler))

  return delta_rot, delta_euler

'''
Get angular rate from two attitudes and time.

Args:
  delta_time: time cost from att1 to att2
  att1, att2: two attitudes
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for input attitudes and output augular rate
Return:
  angular rate from att1 to att2 within time
'''
def angular_rate(delta_time, att1, att2, rot_seq, is_degree):
  delta_rot = delta_att(att1, att2, rot_seq, is_degree)[0]

  delta_rotvec = delta_rot.as_rotvec()
  logger.info('delta rotvec: %s' % delta_rotvec)

  angular_rate = delta_rotvec / delta_time
  logger.info('angular rate(rad): %s' % angular_rate)
  if (is_degree):
    angular_rate = np.rad2deg(angular_rate)
    logger.info('angular rate(deg): %s' % angular_rate)

  return angular_rate
