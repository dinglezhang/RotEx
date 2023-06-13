import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import util
from . import RotEx
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
  So define NED_X_ENU which means to exchange each other, whose rotation sequence is 'ZYX'
'''
EULER_D_FRAME_NED_X_ENU_ZYX = np.array([-90, 180, 0])
EULER_R_FRAME_NED_X_ENU_ZYX = np.deg2rad(EULER_D_FRAME_NED_X_ENU_ZYX)

def att_ned_x_enu(att_old_frame, is_degree):
  euler_frame_ned_x_enu = EULER_R_FRAME_NED_X_ENU_ZYX
  if is_degree:
    euler_frame_ned_x_enu = EULER_D_FRAME_NED_X_ENU_ZYX

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
  logger.info('attitude FRD in NED frame: euler(%s)%s' % (util.get_angular_unit(is_degree), att_ned_2_frd))
  att_enu_2_rfu = att_ned_x_enu(att_ned_2_frd, is_degree)
  logger.info('attitude RFU in ENU frame: euler(%s)%s' % (util.get_angular_unit(is_degree), att_enu_2_rfu))

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
  logger.info('attitude RFU in ENU frame: euler(%s)%s' % (util.get_angular_unit(is_degree), att_enu_2_rfu))
  att_ned_2_frd = att_ned_x_enu(att_enu_2_rfu, is_degree)
  logger.info('attitude FRD in NED frame: euler(%s)%s' % (util.get_angular_unit(is_degree), att_ned_2_frd))

  return att_ned_2_frd

'''
Get attitude from ENU frame to body RFU frame, with right slope angle.
Say a vector like body heading vector, it rotates from north direction to its front direction.

Args:
  heading: body heading vector in body RFU frame
  right_slope_angle: the slope angle of body right direction
  is_degree: True is degree and False is radian for input right_slope_angle and output attitude
Return:
  [0]: rotation from ENU frame to body RFU frame
  [1]: body attitude from ENU frame to body RFU frame
'''
def att_enu_2_rfu(heading, right_slope_angle, is_degree):
  rot = RotEx.from_axisY_2_vector(heading, right_slope_angle, is_degree)

  att_ZYX = rot.as_euler('ZYX', is_degree)
  logger.info('attitude in ZYX sequence: euler(%s)%s\n' % (util.get_angular_unit(is_degree), att_ZYX))

  return rot, att_ZYX

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
  angular_unit = util.get_angular_unit(is_degree)
  logger.info('two attitudes input in %s sequence:' % rot_seq)
  logger.info('euler1(%s)%s' % (angular_unit, att1))
  logger.info('euler2(%s)%s' % (angular_unit, att2))

  rot1 = Rotation.from_euler(rot_seq, att1, is_degree)
  rot2 = Rotation.from_euler(rot_seq, att2, is_degree)

  delta_rot = rot1.inv() * rot2

  delta_euler = delta_rot.as_euler(rot_seq, is_degree)
  logger.info('delta euler in %s sequence: euler(%s)%s' % (rot_seq, angular_unit, delta_euler))

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
  if is_degree:
    angular_rate = np.rad2deg(angular_rate)
    logger.info('angular rate(deg): %s' % angular_rate)

  return angular_rate
