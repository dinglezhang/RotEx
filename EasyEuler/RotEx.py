import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import utils

logger = utils.get_logger()

'''
Get vertial rotation vector with v1 and v2, which norm is rotation angle.

Args:
  v1, v2: two vectors, from v1 to v2
Return:
  vertial rotation vector
'''
def get_vertical_rotvec(v1, v2):
  rot_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
  logger.info('angle: %s' % rot_angle)

  rot_vec = np.cross(v1, v2)
  rot_vec = rot_vec / np.linalg.norm(rot_vec) * rot_angle
  logger.info('vec: %s' % rot_vec)

  return rot_vec

'''
Get rotation from two vectors (from v1 to v2) and self roll angle.

Args:
  v1, v2: two vectors, the rotation is from v1 to v2. The norm could be different between v1 and v2.
  self_roll_angle: self roll angle happend on the vector after rotation from v1 to v2
  is_degree: True is degree and False is radian for input self_roll_angle
Return:
  rotation from v1 to v2 with self roll
'''
def from_v1_2_v2(v1, v2, self_roll_angle, is_degree):
  logger.info('v1: %s v2: %s self_roll_angle(%s): %s' % (v1, v2, utils.get_angular_unit(is_degree), self_roll_angle))

  vertical_rotvec = get_vertical_rotvec(v1, v2)
  rot = Rotation.from_rotvec(vertical_rotvec)

  if self_roll_angle != 0:
    if is_degree:
      self_roll_angle = np.deg2rad(self_roll_angle)

    # after rotation from v1 to v2, v2 has same coordinates as v1 the in new frame. So use v1 to get roll rotation
    rot_roll = Rotation.from_rotvec(v1 / np.linalg.norm(v1) * self_roll_angle)
    rot =  rot* rot_roll

    # the following is an equivalent algorithm with the above
    #rot_roll = Rotation.from_rotvec(v2 / np.linalg.norm(v2) * self_roll_angle)
    #rot =  rot_roll * rot

  return rot

'''
Get rotation from axis Y to a vector and with axis X slope angle.

Args:
  v: a target vector coming from axis Y
  axisX_slope_angle: the slope angle of rotated axis X
  is_degree: True is degree and False is radian for input axisX_slope_angle
Return:
  rotation from axis Y to the vector with axis X slope angle
Raise:
  ValueError: If it is impossible to rotate to the vector with the slope angle
'''
def from_axisY_2_vector(v, axisX_slope_angle, is_degree):
  roll_z = -math.atan2(v[0], v[1])
  roll_x = utils.calc_angle_of_vector_against_XY_plane(v)
  roll_y = 0

  if axisX_slope_angle != 0:
    if is_degree:
      axisX_slope_angle_r = np.deg2rad(axisX_slope_angle)
    else:
      axisX_slope_angle_r = axisX_slope_angle

    sin_roll_y = math.sin(axisX_slope_angle_r) / math.cos(roll_x)
    if sin_roll_y >= -1 and sin_roll_y <= 1:
      roll_y = -math.asin(sin_roll_y)
    else:
      max_slope_angle = abs(math.asin(math.cos(roll_x)))
      if is_degree:
        max_slope_angle = np.rad2deg(max_slope_angle)

      raise ValueError('It is impossible to rotate to the vector with %s slope angle. Possible slope angle should be between [%s, %s] ' % (axisX_slope_angle, -max_slope_angle, max_slope_angle))

  euler_r_ZXY = np.array([roll_z, roll_x, roll_y])
  logger.info('euler(rad) in ZXY sequence: %s' % euler_r_ZXY)
  rot = Rotation.from_euler('ZXY', euler_r_ZXY, False)

  return rot

'''
Get rotation from axis X to a vector and with axis Y slope angle.

Args:
  v: a target vector coming from axis X
  axisY_slope_angle: the slope angle of rotated axis Y
  is_degree: True is degree and False is radian for input axisY_slope_angle
Return:
  rotation from axis X to the vector with axis Y slope angle
Raise:
  ValueError: If it is impossible to rotate to the vector with the slope angle
'''
def from_axisX_2_vector(v, axisY_slope_angle, is_degree):
  roll_z = math.atan2(v[1], v[0])
  roll_y = -utils.calc_angle_of_vector_against_XY_plane(v)
  roll_x = 0

  if axisY_slope_angle != 0:
    if is_degree:
      axisY_slope_angle_r = np.deg2rad(axisY_slope_angle)
    else:
      axisY_slope_angle_r = axisY_slope_angle

    sin_roll_x = math.sin(axisY_slope_angle_r) / math.cos(roll_y)
    if sin_roll_x >= -1 and sin_roll_x <= 1:
      roll_x = math.asin(sin_roll_x)
    else:
      max_slope_angle = abs(math.asin(math.cos(roll_y)))
      if is_degree:
        max_slope_angle = np.rad2deg(max_slope_angle)

      raise ValueError('It is impossible to rotate to the vector with %s slope angle. Possible slope angle should be between [%s, %s] ' % (axisY_slope_angle, -max_slope_angle, max_slope_angle))

  euler_r_ZYX = np.array([roll_z, roll_y, roll_x])
  logger.info('euler(rad) in ZYX sequence: %s' % euler_r_ZYX)
  rot = Rotation.from_euler('ZYX', euler_r_ZYX, False)

  return rot

'''
  World frame definition(xyz coordinates):
    ENU (East, North, Up)
    NED (North, East, Down)

  NED_2_ENU and ENU_2_NED are actually the same rotation.
  So define NED_X_ENU which means to exchange each other, whose rotation sequence is selected as 'ZYX'
'''
EULER_D_NED_X_ENU = np.array([-90, 180, 0])
EULER_R_NED_X_ENU = np.deg2rad(EULER_D_NED_X_ENU)

ROT_NED_X_ENU = Rotation.from_euler('ZYX', EULER_R_NED_X_ENU, False)

'''
Get frame rotation of exchange between NED and ENU since they are same rotation.

Args:
  none
Return:
  frame rotation of exchange between NED and ENU
'''
def ned_x_enu():
  return ROT_NED_X_ENU

'''
Get frame rotation from NED to ENU.

Args:
  none
Return:
  frame rotation from NED to ENU
'''
def from_ned_2_enu():
  return ROT_NED_X_ENU

'''
Get frame rotation from ENU to NED.

Args:
  none
Return:
  frame rotation from ENU to NED
'''
def from_enu_2_ned():
  return ROT_NED_X_ENU

'''
Get delta rotation from rot1 to rot2 in the same frame.
Say an body may rotate to attitude1 by rot1, or to attitude2 by rot2. The function can get how it rotates from attitude1 to attitude2.

Args:
  rot1, rot2: two rotations in the same frame, from rot1 to rot2
  in_original_frame: True is to get rotation in the frame which rot1 and rot2 are in originally.
                     False is to get rotation in the frame which is in after rot1
Return:
  rotation from rot1 to rot2
'''
def get_delta_rot(rot1, rot2, in_original_frame):
  rot1_inv = rot1.inv()
  delta_rot = rot1_inv * rot2

  if in_original_frame:
    delta_rot = get_rot_in_new_frame(delta_rot, rot1_inv)

  return delta_rot

'''
Get rotation in new frame from it in old frame.
The key algorithm is that the rotvec in space has no change. Just to get rotvec in the new frame.

Args:
  rot_in_old_frame: rotation in old frame
  rot_frame_old_2_new: frame rotation from old to new
Return:
  rotation in new frame
'''
def get_rot_in_new_frame(rot_in_old_frame, rot_old_2_new_frame):
  rotvec_in_old_frame = rot_in_old_frame.as_rotvec()
  logger.info('rotvec in old frame: %s' % rotvec_in_old_frame)

  rotvec_in_new_frame = rot_old_2_new_frame.inv().apply(rotvec_in_old_frame)
  logger.info('rotvec in new frame: %s' % rotvec_in_new_frame)

  rot_in_new_frame = Rotation.from_rotvec(rotvec_in_new_frame)

  return rot_in_new_frame

'''
Calculate angular velocity by rotation and delta time.

Args:
  rot: the rotation
  delta_time: time cost for the rotation
  is_degree: True is degree and False is radian for output augular velocity
Return:
  [0]: angular velocity, which is a vecotr
  [1]: angular rate, which is a number
'''
def calc_angular_velocity(rot, delta_time, is_degree):
  rotvec = rot.as_rotvec()
  logger.info('rotvec: %s' % rotvec)

  angular_velocity = rotvec / delta_time
  angular_rate = np.linalg.norm(angular_velocity)
  logger.info('angular velocity(rad): %s' % angular_velocity)
  logger.info('angular rate(rad): %s' % angular_rate)
  if is_degree:
    angular_velocity = np.rad2deg(angular_velocity)
    angular_rate = np.rad2deg(angular_rate)
    logger.info('angular velocity(deg): %s' % angular_velocity)
    logger.info('angular rate(deg): %s' % angular_rate)

  return angular_velocity, angular_rate
