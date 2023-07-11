from scipy.spatial.transform import Rotation

from . import utils
from . import rotex

logger = utils.get_logger()

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

ATT_ROT_SEQ_IN_ENU_FRAME = 'ZXY'
ATT_ROT_SEQ_IN_NED_FRAME = 'ZYX'

'''
Get attitude exchange between ENU and NED.

Args:
  att_in_old_frame: attitude in old frame
  is_degree: True is degree and False is radian for both input and output attitudes
  rot_seq_in_old_frame: rotation sequence in old frame
  rot_seq_in_new_frame: rotation sequence in new frame
Return:
  [0]: rotation in new frame
  [1]: attitude in new frame
'''
def change_frame_enu_x_ned(att_in_old_frame, is_degree, rot_seq_in_old_frame, rot_seq_in_new_frame):
  rot_in_old_frame = Rotation.from_euler(rot_seq_in_old_frame, att_in_old_frame, is_degree)
  rot_in_new_frame = rotex.get_rot_in_new_frame(rot_in_old_frame, rotex.enu_x_ned())
  att_in_new_frame = rot_in_new_frame.as_euler(rot_seq_in_new_frame, is_degree)

  return rot_in_new_frame, att_in_new_frame

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
def change_frame_enu_2_ned(rfu_in_enu_frame, is_degree,
                           rot_seq_in_enu_frame = ATT_ROT_SEQ_IN_ENU_FRAME, rot_seq_in_ned_frame = ATT_ROT_SEQ_IN_NED_FRAME):
  logger.info('attitude RFU(%s) in ENU frame: %s' % (utils.get_angular_unit(is_degree), rfu_in_enu_frame))

  (rot_in_ned_frame, frd_in_ned_frame) = change_frame_enu_x_ned(rfu_in_enu_frame, is_degree, rot_seq_in_enu_frame, rot_seq_in_ned_frame)
  logger.info('attitude FRD(%s) in NED frame: %s' % (utils.get_angular_unit(is_degree), frd_in_ned_frame))

  return rot_in_ned_frame, frd_in_ned_frame

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
def change_frame_ned_2_enu(frd_in_ned_frame, is_degree,
                           rot_seq_in_ned_frame = ATT_ROT_SEQ_IN_NED_FRAME, rot_seq_in_enu_frame = ATT_ROT_SEQ_IN_ENU_FRAME):
  logger.info('attitude FRD(%s) in NED frame: %s' % (utils.get_angular_unit(is_degree), frd_in_ned_frame))

  (rot_in_enu_frame, rfu_in_enu_frame) = change_frame_enu_x_ned(frd_in_ned_frame, is_degree, rot_seq_in_ned_frame, rot_seq_in_enu_frame)
  logger.info('attitude RFU(%s) in ENU frame: %s' % (utils.get_angular_unit(is_degree), rfu_in_enu_frame))

  return rot_in_enu_frame, rfu_in_enu_frame

'''
Get attitude RFU in ENU frame by heading with right slope angle.
Say a vector like body heading vector, it rotates from north direction to its front direction.

Args:
  heading_as_rfu: heading vector as body RFU frame in ENU frame
  right_slope_angle: the slope angle of body right direction
  is_degree: True is degree and False is radian for input right_slope_angle and output attitude
  rot_seq: rotation sequence in ENU frame, ZXY by default
Return:
  [0]: rotation from ENU frame to body RFU frame
  [1]: attitude RFU in ENU frame
'''
def from_heading_in_enu_frame(heading_as_rfu, right_slope_angle, is_degree,
                              rot_seq = ATT_ROT_SEQ_IN_ENU_FRAME):
  rot = rotex.from_y_axis_2_vector(heading_as_rfu, right_slope_angle, is_degree)

  rfu_in_enu_frame = rot.as_euler(rot_seq, is_degree)
  logger.info('attitude RFU(%s) in ENU frame: %s' % (utils.get_angular_unit(is_degree), rfu_in_enu_frame))

  return rot, rfu_in_enu_frame

'''
Get attitude FRD in NED frame by heading with right slope angle.
Say a vector like body heading vector, it rotates from east direction to its front direction.

Args:
  heading_as_frd: heading vector as body FRD frame in NED frame
  right_slope_angle: the slope angle of body right direction
  is_degree: True is degree and False is radian for input right_slope_angle and output attitude
  rot_seq: rotation sequence in NED frame, ZYX by default
Return:
  [0]: rotation from NED frame to body FRD frame
  [1]: attitude FRD in NED frame
'''
def from_heading_in_ned_frame(heading_as_frd, right_slope_angle, is_degree,
                              rot_seq = ATT_ROT_SEQ_IN_NED_FRAME):
  rot = rotex.from_x_axis_2_vector(heading_as_frd, -right_slope_angle, is_degree)

  frd_in_ned_frame = rot.as_euler(rot_seq, is_degree)
  logger.info('attitude FRD(%s) in NED frame: %s' % (utils.get_angular_unit(is_degree), frd_in_ned_frame))

  return rot, frd_in_ned_frame

'''
Get delta euler from att1 to att2 in the same world frame.
Say an body may rotate to att1 or to att2 in the same world frame. The function can get how it rotates from att1 to att2.

Args:
  att1, att2: two attitudes in the same world frame, from att1 to att2
  rot_seq: rotation sequence for the both input attitudes and output delta euler
  is_degree: True is degree and False is radian for input attitudes and output delta attitude
  in_world_frame: True is to get delta euler in the world frame which att1 and att2 are in originally.
                  False is to get delta euler in the world frame which is in after att1
Return:
  [0]: delta rotation from att1 to att2
  [1]: delta euler from att1 to att2
'''
def get_delta_att(att1, att2, rot_seq, is_degree, in_world_frame):
  angular_unit = utils.get_angular_unit(is_degree)
  logger.info('two attitudes(%s) input in %s sequence:' % (angular_unit, rot_seq))
  logger.info('att1: %s' % att1)
  logger.info('att2: %s' % att2)

  rot1 = Rotation.from_euler(rot_seq, att1, is_degree)
  rot2 = Rotation.from_euler(rot_seq, att2, is_degree)

  delta_rot = rotex.get_delta_rot(rot1, rot2, in_world_frame)

  delta_euler = delta_rot.as_euler(rot_seq, is_degree)
  logger.info('delta euler(%s) in %s sequence: %s' % (rot_seq, angular_unit, delta_euler))

  return delta_rot, delta_euler

'''
Calculate angular velocity from two attitudes and time cost.

Args:
  att1, att2: two attitudes
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for input attitudes and output augular rate
  delta_time: time cost from att1 to att2
  in_world_frame: True is to get angular velocity in the world frame which att1 and att2 are in originally.
                  False is to get angular velocity in the world frame which is in after att1
Return:
  angular velocity (vector) from att1 to att2 within time cost
  angular rate (number) from att1 to att2 within time cost
'''
def calc_angular_velocity(att1, att2, rot_seq, is_degree, delta_time, in_world_frame):
  delta_rot = get_delta_att(att1, att2, rot_seq, is_degree, in_world_frame)[0]
  (angular_velocity, angular_rate) = rotex.calc_angular_velocity(delta_rot, delta_time, is_degree)

  return angular_velocity, angular_rate
