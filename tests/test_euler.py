import math
import numpy as np
from scipy.spatial.transform import Rotation

from helpers import util
from . import test_util

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
  angle_unit = util.get_angle_unit(is_degree)
  print('euler(%s) input by %s sequence: %s\n' % (angle_unit, rot_seq, euler))

  rot = Rotation.from_euler(rot_seq, euler, is_degree)

  dcm = rot.as_matrix()
  print('***dcm***\n%s\n' % dcm)

  quat = rot.as_quat()
  print('***quat***\n%s\n' % quat)

  # output rotvec and mrp and test on their consistence
  rotvec = rot.as_rotvec()
  mrp = rot.as_mrp()
  cross_rotvec_mrp = np.cross(rotvec, mrp)
  norm_rotvec = np.linalg.norm(rotvec)
  norm_mrp = np.linalg.norm(mrp)

  result = test_util.get_result(np.allclose(cross_rotvec_mrp, [0, 0, 0]) and    # the two are parallel
                                np.allclose(math.tan(norm_rotvec/4), norm_mrp)) # angle relation
  print('***rotvec and mrp output: %s***' % result)
  print('rotvec: %s angle:        %s' % (rotvec, norm_rotvec))
  print('mrp:    %s tan(angle/4): %s\n' % (mrp, norm_mrp))

  print('***euler by all rotation sequences***')
  for rot_seq in (util.ROTATION_SEQUENCES_INTRINSIC + util.ROTATION_SEQUENCES_EXTRINSIC):
    euler_seq = rot.as_euler(rot_seq, is_degree)
    print('%s: %s' % (rot_seq, euler_seq))

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
  angle_unit = util.get_angle_unit(is_degree)
  print('euler(%s) input by %s sequence: %s\n' % (angle_unit, rot_seq, euler))

  rot_once = Rotation.from_euler(rot_seq, euler, is_degree)
  euler_once = rot_once.as_euler(rot_seq, is_degree)

  rot1 = Rotation.from_euler(rot_seq, [euler[0], 0, 0], is_degree)
  rot2 = Rotation.from_euler(rot_seq, [0, euler[1], 0], is_degree)
  rot3 = Rotation.from_euler(rot_seq, [0, 0, euler[2]], is_degree)
  rot_one_by_one = rot1 * rot2 * rot3
  euler_one_by_one = rot_one_by_one.as_euler(rot_seq, is_degree)

  result = test_util.get_result(np.allclose(euler_once, euler_one_by_one))
  print('***euler(%s) between once and one by one: %s***' % (angle_unit, result))
  print('once: %s' % euler_once)
  print('one by one: %s\n' % euler_one_by_one)

def test_euler_2_x():
  print('============================test euler2x============================')

  euler_d = np.array([20, 1, 5])
  for rot_seq in util.ROTATION_SEQUENCES_INTRINSIC:
    euler_2_x(euler_d, rot_seq, True)
    break # remove it to test all rotation sequences

def test_euler_one_by_one():
  print('============================test euler one by one============================')

  euler_d_input = np.array([20, 10, 15])
  for seq in util.ROTATION_SEQUENCES_INTRINSIC:
    euler_one_by_one(euler_d_input, seq, True)
    break # remove it to test all rotation sequences

def test():
  test_euler_2_x()
  test_euler_one_by_one()
