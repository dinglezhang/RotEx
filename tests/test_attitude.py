import pytest
import math
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, RotationSpline

from RotEx import utils
from RotEx import rotex
from RotEx import attitude

from . import test_rotate_vectors

att_in_frame_1 = \
  [np.array([-45, 0, 0]),
   np.array([0, 45, 0]),
   np.array([0, 0, 45]),
   np.array([-90, 45, 90])]

att_in_frame_2 = \
  [np.array([45, 0, 0]),
   np.array([0, 45, 0]),
   np.array([0, 0, 45]),
   np.array([90, 45, 90])]

heading_samples = \
  [np.array([-1, 1, math.sqrt(2)]),
   np.array([-1, 2, 3]),
   np.array([-1, 2, 0.5])]

slope_angle_samples = [0, 15, 45, -30]

att_d_smaples = \
  [np.array([10, -1, 4]),
   np.array([11, -2, 5]),
   np.array([12, -3, 6])]

rot_seq_samples = ['ZYX', 'zyx']

@pytest.mark.parametrize('rfu_d_in_enu_frame, rfu_d_in_enu_frame_expected', [(a, b) for a, b in zip(att_in_frame_1, att_in_frame_2)])
def test_change_frame_enu_2_ned(rfu_d_in_enu_frame, rfu_d_in_enu_frame_expected):
  (rot_in_ned_frame, frd_d_in_ned_frame) = attitude.change_frame_enu_2_ned(rfu_d_in_enu_frame, True)
  assert_allclose(frd_d_in_ned_frame, rfu_d_in_enu_frame_expected, atol = 1e-8)

@pytest.mark.parametrize('frd_d_in_ned_frame, frd_d_in_ned_frame_expected', [(a, b) for a, b in zip(att_in_frame_2, att_in_frame_1)])
def test_change_frame_ned_2_enu(frd_d_in_ned_frame, frd_d_in_ned_frame_expected):
  (rot_in_enu_frame, rfu_d_in_enu_frame) = attitude.change_frame_ned_2_enu(frd_d_in_ned_frame, True)
  assert_allclose(rfu_d_in_enu_frame, frd_d_in_ned_frame_expected, atol = 1e-8)

@pytest.mark.parametrize('heading_as_rfu', heading_samples)
@pytest.mark.parametrize('right_slope_angle_d', slope_angle_samples)
def test_from_heading_in_enu_frame(heading_as_rfu, right_slope_angle_d):
  is_possible = rotex._analyze_vector_and_angle_against_xy_plane(heading_as_rfu, np.deg2rad(right_slope_angle_d))[0]

  if not is_possible:
    with pytest.raises(ValueError,  match='It is impossible to rotate to the vector'):
      attitude.from_heading_in_enu_frame(heading_as_rfu, right_slope_angle_d, True)
  else:
    (rot, att_d_through_euler) = attitude.from_heading_in_enu_frame(heading_as_rfu, right_slope_angle_d, True)

    # test on heading vector
    heading_start = np.array([0, 1, 0])
    heading_end_expected = heading_as_rfu / np.linalg.norm(heading_as_rfu)
    test_rotate_vectors.test_rotate_vectors_once(heading_start, att_d_through_euler, attitude.ATT_ROT_SEQ_IN_ENU_FRAME,
                                                 heading_end_expected, False)

    # test on right slope angle by right direction
    right_start = np.array([1, 0, 0])
    right_end = rot.apply(right_start)
    right_slope_angle_result = utils.calc_angle_between_vector_and_xy_plane(right_end, True)

    assert_allclose(right_slope_angle_result, right_slope_angle_d, atol = 1e-8)

@pytest.mark.parametrize('heading_as_frd', heading_samples)
@pytest.mark.parametrize('right_slope_angle_d', slope_angle_samples)
def test_from_heading_in_ned_frame(heading_as_frd, right_slope_angle_d):
  is_possible = rotex._analyze_vector_and_angle_against_xy_plane(heading_as_frd, np.deg2rad(right_slope_angle_d))[0]

  if not is_possible:
    with pytest.raises(ValueError,  match='It is impossible to rotate to the vector'):
      attitude.from_heading_in_ned_frame(heading_as_frd, right_slope_angle_d, True)
  else:
    (rot, att_d_through_euler) = attitude.from_heading_in_ned_frame(heading_as_frd, right_slope_angle_d, True)

    # test on heading vector
    heading_start = np.array([1, 0, 0])
    heading_end_expected = heading_as_frd / np.linalg.norm(heading_as_frd)
    test_rotate_vectors.test_rotate_vectors_once(heading_start, att_d_through_euler, attitude.ATT_ROT_SEQ_IN_NED_FRAME,
                                                 heading_end_expected, False)

    # test on right slope angle by right direction
    right_start = np.array([0, 1, 0])
    right_end = rot.apply(right_start)
    right_slope_angle_result = -utils.calc_angle_between_vector_and_xy_plane(right_end, True)

    assert_allclose(right_slope_angle_result, right_slope_angle_d, atol = 1e-8)

@pytest.mark.parametrize('att_d_1', att_d_smaples)
@pytest.mark.parametrize('att_d_2', att_d_smaples)
@pytest.mark.parametrize('rot_seq', rot_seq_samples)
@pytest.mark.parametrize('in_world_frame', [True, False])
def test_get_delta_att(att_d_1, att_d_2, rot_seq, in_world_frame, vector_samples):
  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rot2 = Rotation.from_euler(rot_seq, att_d_2, True)

  if in_world_frame:
    vectors_by_rot1 = rot1.apply(vector_samples)
    vectors_by_rot2 = rot2.apply(vector_samples)
  else:
    vectors_by_rot1 = vector_samples # they are original coordinates in rot1 frame
    vectors_by_rot2 = rot2.apply(vector_samples)
    vectors_by_rot2 = rot1.inv().apply(vectors_by_rot2)

  (delta_rot, delta_euler_d) = attitude.get_delta_att(att_d_1, att_d_2, rot_seq, True, in_world_frame)
  vectors_by_delta_rot = delta_rot.apply(vectors_by_rot1)

  assert_allclose(vectors_by_rot2, vectors_by_delta_rot, atol = 1e-8)

@pytest.mark.parametrize('att_d_1', att_d_smaples)
@pytest.mark.parametrize('factor', [1.1, 1.2])
@pytest.mark.parametrize('rot_seq', rot_seq_samples)
def test_linear_delta_att(att_d_1, factor, rot_seq):
  # calculate att_d_2 from att_d_1 and factor (linear change on rotvec)
  rot1 = Rotation.from_euler(rot_seq, att_d_1, True)
  rotvec1 = rot1.as_rotvec()
  rotvec2 = rotvec1 * factor
  rot2 = Rotation.from_rotvec(rotvec2)
  att_d_2 = rot2.as_euler(rot_seq, True)

  # calculate delta_att by attitude.delta_att()
  (delta_rot, delta_euler_d) = attitude.get_delta_att(att_d_1, att_d_2, rot_seq, True, False)

  # calcaulate delta_att_linear by a special way for linear rotvec change
  delta_rotvec_linear = rotvec2 - rotvec1
  delta_rot_linear = Rotation.from_rotvec(delta_rotvec_linear)
  delta_euler_d_linear = delta_rot_linear.as_euler(rot_seq, True)

  assert_allclose(delta_euler_d, delta_euler_d_linear)

@pytest.mark.parametrize('att_d_0', att_d_smaples)
@pytest.mark.parametrize('att_d_1', att_d_smaples)
@pytest.mark.parametrize('att_d_2', att_d_smaples)
@pytest.mark.parametrize('rot_seq', rot_seq_samples)
@pytest.mark.parametrize('delta_time', [2, 3])
def test_calc_angular_velocity_acceleration(att_d_0, att_d_1, att_d_2, rot_seq, delta_time):
  angular_velocity_1 = attitude.calc_angular_velocity(att_d_0, att_d_1, rot_seq, True, delta_time, False)[0]
  angular_velocity_2 = attitude.calc_angular_velocity(att_d_1, att_d_2, rot_seq, True, delta_time, False)[0]
  angular_acceleration = rotex.calc_angular_acceleration(angular_velocity_2, angular_velocity_1, delta_time)[0]

  times = [0, delta_time, delta_time * 2]
  angles = [att_d_0, att_d_1, att_d_2]
  rotations = Rotation.from_euler(rot_seq, angles, True)
  spline = RotationSpline(times, rotations)

  times = [0, delta_time * 2]
  angular_velocity_spline = spline(times, 1)
  angular_velocity_spline = np.rad2deg(angular_velocity_spline)
  angular_acceleration_spline = spline(times, 2)
  angular_acceleration_spline = np.rad2deg(angular_acceleration_spline)

  assert_allclose(angular_velocity_1, angular_velocity_spline[0], atol = 1e-8)
  assert_allclose(angular_velocity_2, angular_velocity_spline[1], atol = 1e-8)
  assert_allclose(angular_acceleration, (angular_acceleration_spline[0] + angular_acceleration_spline[1])/2, atol = 1e-6)
