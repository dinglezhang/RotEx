import numpy as np
import quaternion
from scipy.spatial.transform import Rotation

from . import common

'''
to rotate by 4 ways:
  Rotation
  dcm by scipy.Rotation
  dcm by attitude
  quaternion
they should get the same results
'''
def vectors_rotation(vectors_input, euler_d_input, rot_seq, times = 1, is_extrinsic = False, on_frame = False):
  times_str = ''
  if times > 1:
    times_str = '%s times composed ' % times
  way_str = 'intrinsicly'
  if is_extrinsic:
    way_str = 'extrinsicly'
  on_str = 'on vector'
  if on_frame:
    on_str = 'on frame'
  print('~~~~~~~~~~vectors rotation %s%s %s~~~~~~~~~~' % (times_str, way_str, on_str))

  print('vector input:\n%s' % vectors_input)
  euler_r_input = euler_d_input * common.D2R
  print('euler input by %s sequence: %s (%s)\n' % (rot_seq, euler_d_input, euler_r_input))

  rot_seq_scipy = rot_seq.upper() # intrinsic
  if is_extrinsic:
    rot_seq_scipy = rot_seq     # extrinsic
  rot_once = Rotation.from_euler(rot_seq_scipy, euler_r_input)

  dcm_scipy_once = rot_once.as_matrix()

  quat_vectors_input = quaternion.from_vector_part(vectors_input)
  quat_once = quaternion.from_float_array(rot_once.as_quat()[[3, 0, 1, 2]])

  rot = rot_once
  dcm_scipy = dcm_scipy_once
  quat = quat_once
  for i in range(1, times):
    rot = rot * rot_once
    dcm_scipy = dcm_scipy.dot(dcm_scipy_once)
    quat = quat * quat_once

  vectors_rotated_scipy = rot.apply(vectors_input, on_frame)

  if (not on_frame):  # to rotate vectors
    vectors_rotated_scipy_dcm = (dcm_scipy.dot(vectors_input.T)).T  # left multiplication to rotate vectors
    vectors_rotated_quat = quaternion.as_vector_part(quat * quat_vectors_input * quat.conjugate())
  else:  # to rotate frame
    vectors_rotated_scipy_dcm = vectors_input.dot(dcm_scipy)        # right multiplication to rotate frame
    vectors_rotated_quat = quaternion.as_vector_part(quat.conjugate() * quat_vectors_input * quat)

  is_passed = False
  if np.allclose(vectors_rotated_scipy, vectors_rotated_scipy_dcm) and\
    np.allclose(vectors_rotated_scipy, vectors_rotated_quat):
      is_passed = True
  result = common.get_result(is_passed)

  print('***vetors rotated: %s***' % result)
  print('Rotation:\n%s' % vectors_rotated_scipy)
  print('scipy dcm:\n%s' % vectors_rotated_scipy_dcm)
  print('quaternion:\n%s' % vectors_rotated_quat)
  print()

  return vectors_rotated_scipy
