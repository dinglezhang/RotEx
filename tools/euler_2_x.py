import sys
import math
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

sys.path.append('..\\')
from EasyEuler import utils

'''
Print all kinds of expression of a rotation by rotation.

Args:
  rot: rotation
  is_degree: True is degree and False is radian for output eulers
Return:
  None
'''
def rot_2_x(rot, is_degree):
  dcm = rot.as_matrix()
  print('dcm: %s' % dcm)

  quat = rot.as_quat()
  print('quat: %s' % quat)

  # output rotvec and mrp and test on their consistence
  rotvec = rot.as_rotvec()
  mrp = rot.as_mrp()
  cross_rotvec_mrp = np.cross(rotvec, mrp)
  norm_rotvec = np.linalg.norm(rotvec)
  norm_mrp = np.linalg.norm(mrp)

  assert_allclose(cross_rotvec_mrp, [0, 0, 0], atol=1e-8)        # the two are parallel
  assert_allclose(math.tan(norm_rotvec/4), norm_mrp)  # angle relation

  if is_degree:
    norm_rotvec = np.rad2deg(norm_rotvec)
  print('rotvec: %s angle:        %s' % (rotvec, norm_rotvec))

  if is_degree:
    norm_mrp = np.rad2deg(norm_mrp)
  print('mrp: %s tan(angle/4): %s' % (mrp, norm_mrp))

  for rot_seq in (utils.ROTATION_SEQUENCES_INTRINSIC + utils.ROTATION_SEQUENCES_EXTRINSIC):
    euler_seq = rot.as_euler(rot_seq, is_degree)
    print('euler(%s): %s' % (rot_seq, euler_seq))
  print()

'''
Print all kinds of expression of a rotation by euler angles.

Args:
  euler: euler angles to rotate
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for both input and output eulers
Return:
  None
'''
def euler_2_x(euler, rot_seq, is_degree):
  print('euler(%s) input in %s sequence: %s' % (utils.get_angular_unit(is_degree), rot_seq, euler))

  rot = Rotation.from_euler(rot_seq, euler, is_degree)
  rot_2_x(rot, is_degree)

'''
Rotate by euler angles once or 3 times one by one, and test if they get the same results

Args:
  euler: euler angles to rotate
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for input euler
Return:
  None
'''
def euler_one_by_one(euler, rot_seq, is_degree):
  print('euler(%s) input in %s sequence: %s' % (utils.get_angular_unit(is_degree), rot_seq, euler))

  rot_once = Rotation.from_euler(rot_seq, euler, is_degree)
  euler_once = rot_once.as_euler(rot_seq, is_degree)

  rot0 = Rotation.from_euler(rot_seq[0], euler[0], is_degree)
  rot1 = Rotation.from_euler(rot_seq[1], euler[1], is_degree)
  rot2 = Rotation.from_euler(rot_seq[2], euler[2], is_degree)
  rot_one_by_one = rot0 * rot1 * rot2
  euler_one_by_one = rot_one_by_one.as_euler(rot_seq, is_degree)

  assert_allclose(euler_one_by_one, euler_once)

  print('euler0(%s): %s' % (rot_seq[0], rot0.as_euler(rot_seq, is_degree)))
  print('euler1(%s): %s' % (rot_seq[1], rot1.as_euler(rot_seq, is_degree)))
  print('euler2(%s): %s\n' % (rot_seq[2], rot2.as_euler(rot_seq, is_degree)))

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  euler_d = np.array([20, 1, 5])
  for rot_seq in utils.ROTATION_SEQUENCES_INTRINSIC:
    euler_2_x(euler_d, rot_seq, True)
    euler_one_by_one(euler_d, rot_seq, True)
    break # remove it to test all rotation sequences
