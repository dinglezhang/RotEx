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
Analyze rotation from x or y axis to a vector and with an angle between cross vector and xy plane, against xy plane
All input and output angle is radian unit

Args:
  v: a target vector
  angle_crossv_and_xy_plane: angle between cross vector of the target vector and xy plane
Return:
  [0]: True or False for the possibility
  [1]: angle between vector and xy plane
  [2]: angle of self roll of the vector, which is with value only if the possibility is True
  [3]: max possible angle between cross vector and xy plane, which is with value only if the possibility is False
'''
def _analyze_vector_and_angle_against_xy_plane(v, angle_crossv_and_xy_plane):
  is_possible = False

  angle_v_and_xy_plane = utils.calc_angle_between_vector_and_xy_plane(v)

  sin_angle_self_roll = math.sin(angle_crossv_and_xy_plane) / math.cos(angle_v_and_xy_plane)

  angle_self_roll = None
  if sin_angle_self_roll >= -1 and sin_angle_self_roll <= 1:
    is_possible = True
    angle_self_roll = math.asin(sin_angle_self_roll)

  max_angle_crossv_and_xy_plane = None
  if not is_possible:
    max_angle_crossv_and_xy_plane = abs(math.asin(math.cos(angle_v_and_xy_plane)))

  return is_possible, angle_v_and_xy_plane, angle_self_roll, max_angle_crossv_and_xy_plane

'''
Get rotation from x-axis to a vector and with y-axis slope angle.

Args:
  v: a target vector coming from x-axis
  y-axis_slope_angle: the slope angle of rotated y-axis
  is_degree: True is degree and False is radian for input y-axis_slope_angle
Return:
  rotation from x-axis to the vector with y-axis slope angle
Raise:
  ValueError: If it is impossible to rotate to the vector with the slope angle
'''
def from_x_axis_2_vector(v, y_axis_slope_angle, is_degree):
  roll_z = math.atan2(v[1], v[0])
  roll_y = 0
  roll_x = 0

  if is_degree:
    y_axis_slope_angle = np.deg2rad(y_axis_slope_angle)

  (is_possible, roll_y, angle_self_roll, max_slope_angle) = _analyze_vector_and_angle_against_xy_plane(v, y_axis_slope_angle)
  if is_possible:
    roll_x = angle_self_roll
  else:
    if is_degree:
      max_slope_angle = np.rad2deg(max_slope_angle)

    raise ValueError('It is impossible to rotate to the vector with %s slope angle. Possible slope angle should be between [%s, %s] ' % (y_axis_slope_angle, -max_slope_angle, max_slope_angle))

  euler_r_ZYX = np.array([roll_z, -roll_y, roll_x])
  logger.info('euler(rad) in ZYX sequence: %s' % euler_r_ZYX)
  rot = Rotation.from_euler('ZYX', euler_r_ZYX, False)

  return rot

'''
Get rotation from y-axis to a vector and with x-axis slope angle.

Args:
  v: a target vector coming from y-axis
  x-axis_slope_angle: the slope angle of rotated x-axis
  is_degree: True is degree and False is radian for input x-axis_slope_angle
Return:
  rotation from y-axis to the vector with x-axis slope angle
Raise:
  ValueError: If it is impossible to rotate to the vector with the slope angle
'''
def from_y_axis_2_vector(v, x_axis_slope_angle, is_degree):
  roll_z = -math.atan2(v[0], v[1])
  roll_x = 0
  roll_y = 0

  if is_degree:
    x_axis_slope_angle = np.deg2rad(x_axis_slope_angle)

  (is_possible, roll_x, angle_self_roll, max_slope_angle) = _analyze_vector_and_angle_against_xy_plane(v, x_axis_slope_angle)

  if is_possible:
    roll_y = -angle_self_roll
  else:
    if is_degree:
      max_slope_angle = np.rad2deg(max_slope_angle)

    raise ValueError('It is impossible to rotate to the vector with %s slope angle. Possible slope angle should be between [%s, %s] ' % (x_axis_slope_angle, -max_slope_angle, max_slope_angle))

  euler_r_ZXY = np.array([roll_z, roll_x, roll_y])
  logger.info('euler(rad) in ZXY sequence: %s' % euler_r_ZXY)
  rot = Rotation.from_euler('ZXY', euler_r_ZXY, False)

  return rot

'''
  World frame definition(xyz coordinates):
    ENU (East, North, Up)
    NED (North, East, Down)

  ENU_2_NED and NED_2_ENU are actually the same rotation.
  So define ENU_X_NED which means to exchange each other, whose rotation sequence is selected as 'ZYX'
'''
EULER_D_ENU_X_NED = np.array([-90, 180, 0])
EULER_R_ENU_X_NED = np.deg2rad(EULER_D_ENU_X_NED)

ROT_ENU_X_NED = Rotation.from_euler('ZYX', EULER_R_ENU_X_NED, False)

'''
Get frame rotation of exchange between ENU and NED since they are same rotation.

Args:
  none
Return:
  frame rotation of exchange between ENU and NED
'''
def enu_x_ned():
  return ROT_ENU_X_NED

'''
Get frame rotation from ENU to NED.

Args:
  none
Return:
  frame rotation from ENU to NED
'''
def from_enu_2_ned():
  return ROT_ENU_X_NED

'''
Get frame rotation from NED to ENU.

Args:
  none
Return:
  frame rotation from NED to ENU
'''
def from_ned_2_enu():
  return ROT_ENU_X_NED

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
Calculate rotation angular displacement.

Args:
  rot: the rotation
  is_degree: True is degree and False is radian for output augular velocity
Return:
  [0]: rotation angular vector
  [1]: rotation angular scalar
'''
def calc_angular_displacement(rot, is_degree):
  rotvec = rot.as_rotvec()
  logger.info('rotvec: %s' % rotvec)

  angular_vector = rotvec
  angular_scalar = np.linalg.norm(angular_vector)
  logger.info('rotation angular vector(rad): %s' % angular_vector)
  logger.info('rotation angular scalar(rad): %s' % angular_scalar)
  if is_degree:
    angular_vector = np.rad2deg(angular_vector)
    angular_scalar = np.rad2deg(angular_scalar)
    logger.info('rotation angular vector(deg): %s' % angular_vector)
    logger.info('rotation angular scalar(deg): %s' % angular_scalar)

  return angular_vector, angular_scalar

'''
Calculate angular velocity by rotation and delta time.

Args:
  rot: the rotation
  delta_time: time cost for the rotation
  is_degree: True is degree and False is radian for output augular velocity
Return:
  [0]: angular velocity, which is a vecotr
  [1]: angular rate, which is a scalar
'''
def calc_angular_velocity(rot, delta_time, is_degree):
  (angular_vector, angular_scalar) = calc_angular_displacement(rot, is_degree)

  angular_velocity = angular_vector / delta_time
  angular_rate = angular_scalar / delta_time
  angular_unit = utils.get_angular_unit(is_degree)
  logger.info('angular velocity(%s): %s' % (angular_unit, angular_velocity))
  logger.info('angular rate(%s): %s' % (angular_unit, angular_rate))

  return angular_velocity, angular_rate

'''
Calculate angular acceleration by two angular velocities and delta time.
[ToDo] So far it is just to get difference between two angular velocities and divide by delta time, temporarily.
       Need to find a good way to decribe rot with changing angular velocity, then to calculate the angular acceleration.

Args:
  angular_velocity_1, angular_velocity_2: two angular velocities from 1 to 2
  delta_time: time cost for the change of angular_velocity
Return:
  [0]: angular acceleration vector
  [1]: angular acceleration scalar
  whether accelerationis degree or not dependends on the input
'''
def calc_angular_acceleration(angular_velocity_1, angular_velocity_2, delta_time):
  angular_acceleration_vector = (angular_velocity_2 - angular_velocity_1) / delta_time
  angular_acceleration_scalar = np.linalg.norm(angular_acceleration_vector)

  return angular_acceleration_vector, angular_acceleration_scalar

'''
Calculate linear displacement for vectors by rotation.

Args:
  rot: the rotation
  vectors: vectors to be rotated
Return:
  [0]: linear displacement vectors
  [1]: linear displacement scalars
'''
def calc_linear_displacement(rot, vectors):
  angular_vector = calc_angular_displacement(rot, False)[0]

  linear_displacement_vectors = np.cross(angular_vector, vectors)
  linear_displacement_scalars = np.linalg.norm(linear_displacement_vectors, axis = 1)

  return linear_displacement_vectors, linear_displacement_scalars

'''
Calculate linear velocity for vectors by rotation and delta time.

Args:
  rot: the rotation
  vectors: vectors to be rotated
  delta_time: time cost for the rotation
Return:
  [0]: linear velocities, which are vectors
  [1]: linear rates, which are scalars
'''
def calc_linear_velocity(rot, vectors, delta_time):
  angular_velocity = calc_angular_velocity(rot, delta_time, False)[0]

  linear_velocities = np.cross(angular_velocity, vectors)
  linear_rates = np.linalg.norm(linear_velocities, axis = 1)

  return linear_velocities, linear_rates

'''
Calculate centripetal acceleration for vectors by rotation and delta time.

Args:
  rot: the rotation
  vectors: vectors to be rotated
  delta_time: time cost for the rotation
Return:
  [0]: centripetal acceleration vectors
  [1]: centripetal acceleration scalars
'''
def calc_centripetal_acceleration(rot, vectors, delta_time):
  angular_velocity = calc_angular_velocity(rot, delta_time, False)[0]

  centripetal_acceleration_vectors = np.cross(angular_velocity, np.cross(angular_velocity, vectors))
  centripetal_acceleration_scalars = np.linalg.norm(centripetal_acceleration_vectors, axis = 1)

  return centripetal_acceleration_vectors, centripetal_acceleration_scalars
