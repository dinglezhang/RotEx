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

  Attitude always uses yaw-pitch-roll as rotation sequence.
    yaw-pitch-roll is 'ZXY' as attitude RFU in ENU frame
    yaw-pitch-roll is 'ZYX' as attitude FRD in NED frame
'''

'''
Get attitude exchange between NED and ENU.

Args:
  att_in_old_frame: attitude in old frame
  is_degree: True is degree and False is radian for both input and output attitudes
  rot_seq_in_old_frame: rotation sequence in old frame
  rot_seq_in_new_frame: rotation sequence in new frame
Return:
  [0]: rotation in new frame
  [1]: attitude in new frame
'''
def change_frame_ned_x_enu(att_in_old_frame, is_degree, rot_seq_in_old_frame, rot_seq_in_new_frame):
  rot_in_old_frame = Rotation.from_euler(rot_seq_in_old_frame, att_in_old_frame, is_degree)
  rot_in_new_frame = RotEx.get_rot_in_new_frame(rot_in_old_frame, RotEx.ned_x_enu())
  att_in_new_frame = rot_in_new_frame.as_euler(rot_seq_in_new_frame, is_degree)

  return rot_in_new_frame, att_in_new_frame

'''
Get attitude RFU in ENU frame from it(FRD) in NED frame.

Args:
  frd_in_ned_frame: attitude FRD in NED frame
  is_degree: True is degree and False is radian for both input and output attitudes
  rot_seq_in_ned_frame: rotation sequence in NED frame, ZYX by default
  rot_seq_in_enu_frame: rotation sequence in ENU frame, ZXY by default
Return:
  [0]: rotation in ENU frame
  [1]: attitude RFU in ENU frame
'''
def change_frame_ned_2_enu(frd_in_ned_frame, is_degree, rot_seq_in_ned_frame = 'ZYX', rot_seq_in_enu_frame = 'ZXY'):
  logger.info('attitude FRD(%s) in NED frame: %s' % (util.get_angular_unit(is_degree), frd_in_ned_frame))

  (rot_in_enu_frame, rfu_in_enu_frame) = change_frame_ned_x_enu(frd_in_ned_frame, is_degree, rot_seq_in_ned_frame, rot_seq_in_enu_frame)
  logger.info('attitude RFU(%s) in ENU frame: %s' % (util.get_angular_unit(is_degree), rfu_in_enu_frame))

  return rot_in_enu_frame, rfu_in_enu_frame

'''
Get attitude FRD in NED frame from it(RFU) in ENU frame.

Args:
  rfu_in_enu_frame: attitude RFU in ENU frame
  is_degree: True is degree and False is radian for both input and output attitudes
  rot_seq_in_enu_frame: rotation sequence in ENU frame, ZXY by default
  rot_seq_in_ned_frame: rotation sequence in NED frame, ZYX by default
Return:
  [0]: rotation in NED frame
  [1]: attitude FRD in NED frame
'''
def change_frame_enu_2_ned(rfu_in_enu_frame, is_degree, rot_seq_in_enu_frame = 'ZXY', rot_seq_in_ned_frame = 'ZYX'):
  logger.info('attitude RFU(%s) in ENU frame: %s' % (util.get_angular_unit(is_degree), rfu_in_enu_frame))

  (rot_in_ned_frame, frd_in_ned_frame) = change_frame_ned_x_enu(rfu_in_enu_frame, is_degree, rot_seq_in_enu_frame, rot_seq_in_ned_frame)
  logger.info('attitude FRD(%s) in NED frame: %s' % (util.get_angular_unit(is_degree), frd_in_ned_frame))

  return rot_in_ned_frame, frd_in_ned_frame

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
Get delta euler from att1 to att2 in the same world frame.
Say an body may rotate to att1 or to att2 in the same world frame. The function can get how it rotates from att1 to att2.

Args:
  att1, att2: two attitudes in the same world frame, from att1 to att2
  rot_seq: rotation sequence for the both attitudes
  is_degree: True is degree and False is radian for input attitudes and output delta attitude
  in_world_frame: True is to get delta euler in the world frame which att1 and att2 are in originally.
                  False is to get delta euler in the world frame which is in after att1
Return:
  [0]: delta rotation from att1 to att2
  [1]: delta euler from att1 to att2
'''
def get_delta_att(att1, att2, rot_seq, is_degree, in_world_frame):
  angular_unit = util.get_angular_unit(is_degree)
  logger.info('two attitudes(%s) input in %s sequence:' % (angular_unit, rot_seq))
  logger.info('att1: %s' % att1)
  logger.info('att2: %s' % att2)

  rot1 = Rotation.from_euler(rot_seq, att1, is_degree)
  rot2 = Rotation.from_euler(rot_seq, att2, is_degree)

  delta_rot = RotEx.get_delta_rot(rot1, rot2, in_world_frame)

  delta_euler = delta_rot.as_euler(rot_seq, is_degree)
  logger.info('delta euler(%s) in %s sequence: %s' % (rot_seq, angular_unit, delta_euler))

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
  delta_rot = get_delta_att(att1, att2, rot_seq, is_degree, False)[0]

  delta_rotvec = delta_rot.as_rotvec()
  logger.info('delta rotvec: %s' % delta_rotvec)

  angular_rate = delta_rotvec / delta_time
  logger.info('angular rate(rad): %s' % angular_rate)
  if is_degree:
    angular_rate = np.rad2deg(angular_rate)
    logger.info('angular rate(deg): %s' % angular_rate)

  return angular_rate
