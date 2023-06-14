import numpy as np
from scipy.spatial.transform import Rotation

from . import util
from . import RotEx

logger = util.get_logger()

'''
  Attitude is a set of euler angles how a body intrinsic rotates from the world frame to its current body frame.

  World frame (xyz coordinates):
    ENU (East, North, Up)
    NED (North, East, Down)

  Body frame (xyz coordinates):
    RFU (Right, Front, Up)
    FRD (Front, Right, Down)
'''

'''
Get attitude RFU in ENU frame from it(FRD) in NED frame, or
get attitude FRD in NED frame from it(RFU) in ENU frame.

Args:
  att_in_old_frame: attitude in old frame
  is_degree: True is degree and False is radian for both input and output attitudes
Return:
  attitude in new frame
'''
def from_ned_x_enu_frame(att_in_old_frame, is_degree):
  rot_seq = 'ZYX'

  rot_in_old_frame = Rotation.from_euler(rot_seq, att_in_old_frame, is_degree)
  rot_in_new_frame = RotEx.from_rot_in_new_frame(rot_in_old_frame, RotEx.from_ned_x_enu())
  att_in_new_frame = rot_in_new_frame.as_euler(rot_seq, is_degree)

  return att_in_new_frame

'''
Get attitude RFU in ENU frame from it(FRD) in NED frame.

Args:
  frd_in_ned_frame: attitude FRD in NED frame
  is_degree: True is degree and False is radian for both input and output attitudes
Return:
  attitude RFU in ENU frame
'''
def from_ned_2_enu_frame(frd_in_ned_frame, is_degree):
  logger.info('attitude FRD(%s) in NED frame: %s' % (util.get_angular_unit(is_degree), frd_in_ned_frame))

  rfu_in_enu_frame = from_ned_x_enu_frame(frd_in_ned_frame, is_degree)
  logger.info('attitude RFU(%s) in ENU frame: %s' % (util.get_angular_unit(is_degree), rfu_in_enu_frame))

  return rfu_in_enu_frame

'''
Get attitude FRD in NED frame from it(RFU) in ENU frame.

Args:
  rfu_in_enu_frame: attitude RFU in ENU frame
  is_degree: True is degree and False is radian for both input and output attitudes
Return:
  attitude FRD in NED frame
'''
def from_enu_2_ned_frame(rfu_in_enu_frame, is_degree):
  logger.info('attitude RFU(%s) in ENU frame: %s' % (util.get_angular_unit(is_degree), rfu_in_enu_frame))

  frd_in_ned_frame = from_ned_x_enu_frame(rfu_in_enu_frame, is_degree)
  logger.info('attitude FRD(%s) in NED frame: %s' % (util.get_angular_unit(is_degree), frd_in_ned_frame))

  return frd_in_ned_frame

'''
Get attitude RFU in ENU frame by heading with right slope angle.
Say a vector like body heading vector, it rotates from north direction to its front direction.

Args:
  heading_as_rfu: heading vector as body RFU frame in ENU frame
  right_slope_angle: the slope angle of body right direction
  is_degree: True is degree and False is radian for input right_slope_angle and output attitude
Return:
  [0]: rotation from ENU frame to body RFU frame
  [1]: attitude RFU in ENU frame
'''
def from_heading_in_enu_frame(heading_as_rfu, right_slope_angle, is_degree):
  rot = RotEx.from_axisY_2_vector(heading_as_rfu, right_slope_angle, is_degree)

  rfu_in_enu_frame = rot.as_euler('ZYX', is_degree)
  logger.info('attitude RFU(%s) in ENU frame: %s' % (util.get_angular_unit(is_degree), rfu_in_enu_frame))

  return rot, rfu_in_enu_frame

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
