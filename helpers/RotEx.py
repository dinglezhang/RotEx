import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import util

logger = util.get_logger()

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
Get rotation from two vectors and self roll angle.

Args:
  v1, v2: two vectors, the rotation is from v1 to v2. The norm could be different between v1 and v2.
  self_roll_angle: self roll angle happend on the vector after rotation from v1 to v2
  is_degree: True is degree and False is radian for input self_roll_angle
Return:
  rotation from v1 to v2 with self roll
'''
def from_two_vectors(v1, v2, self_roll_angle, is_degree):
  logger.info('v1: %s v2: %s self_roll_angle(%s): %s' % (v1, v2, util.get_angular_unit(is_degree), self_roll_angle))

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
[ToDo] change axisY to a parameter and support axisX and axisZ

Args:
  v: a target vector coming from axis Y
  axisX_slope_angle: the slope angle of rotated axis X
  is_degree: True is degree and False is radian for input axisX_slope_angle
Return:
  rotation from axis Y to the vector with axis X slope angle
'''
def from_axisY_2_vector(v, axisX_slope_angle, is_degree):
  x = v[0]
  y = v[1]
  z = v[2]
  roll_z = math.atan2(-x, y)
  roll_x = math.atan2(z, math.sqrt(x * x + y * y))
  roll_y = 0

  if axisX_slope_angle != 0:
    if is_degree:
      axisX_slope_angle = np.deg2rad(axisX_slope_angle)
    roll_y = -math.asin(math.sin(axisX_slope_angle) / math.cos(roll_x))

  euler_r_ZXY = np.array([roll_z, roll_x, roll_y])
  logger.info('euler in ZXY sequence: euler(rad)%s' % np.rad2deg(euler_r_ZXY))
  rot = Rotation.from_euler('ZXY', euler_r_ZXY, False)

  return rot
