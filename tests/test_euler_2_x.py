import math
import numpy as np
from scipy.spatial.transform import Rotation

from helpers import common

'''
output all kinds of expression of a rotation expressed by euler
as well as tests on the following functions provided by attitude
  euler2dcm()
  euler2quat()
  dcm2euler() and quat2euler() by all rotation sequneces. (looks like euler to euler from input)
  dcm2quat() and quat2dcm()
'''
def euler_2_x(euler_d_input, rot_seq):
  print('~~~~~~~~~~euler2x~~~~~~~~~~')

  euler_r_input = euler_d_input * common.D2R
  print('euler input by %s sequence intrinsicly:' % rot_seq)
  print('%s (%s)\n' % (euler_d_input, euler_r_input))

  rot = Rotation.from_euler(rot_seq.upper(), euler_r_input) # in body frame (intrinsic)

  # output dcm, as well as test on euler2dcm()
  dcm_scipy = rot.as_matrix()
  print('scipy: \n%s' % dcm_scipy)

  # output quat, as well as test on euler2quat()
  quat_scipy = rot.as_quat()
  print('scipy: \n%s' % quat_scipy)

  # output euler for all kinds of rotation sequences intrinsicly, as well as test on dcm2euler() and quat2euler()
  for rot_seq in common.ROTATION_SEQUENCES:
    euler_r_scipy = rot.as_euler(rot_seq.upper())
    euler_d_scipy = euler_r_scipy * common.R2D
    print('scipy:         %s (%s)' % (euler_d_scipy, euler_r_scipy))

  # output rotvec and mrp by scipy and test on their consistence
  rotvec_scipy = rot.as_rotvec()
  mrp_scipy = rot.as_mrp()
  cross_rotvec_mrp = np.cross(rotvec_scipy, mrp_scipy)
  norm_rotvec = np.linalg.norm(rotvec_scipy)
  norm_mrp = np.linalg.norm(mrp_scipy)

  result = common.get_result(np.allclose(cross_rotvec_mrp, [0, 0, 0]) and    # the two are parallel
                             np.allclose(math.tan(norm_rotvec/4), norm_mrp)) # angle relation
  print('***scipy rotvec and mrp output: %s***' % result)
  print('rotvec: %s angle:        %s' % (rotvec_scipy, norm_rotvec*common.R2D))
  print('mrp:    %s tan(angle/4): %s\n' % (mrp_scipy, norm_mrp))

  # output eulers for all kinds of rotation sequences by scipy extrinsicly
  print('***scipy euler output extrinsicly***')
  for rot_seq in common.ROTATION_SEQUENCES:
    euler_r_scipy = rot.as_euler(rot_seq)
    euler_d_scipy = euler_r_scipy * common.R2D
    print('%s: %s (%s)' % (rot_seq, euler_d_scipy, euler_r_scipy))
  print()

'''
to rotate by:
  euler angles once
  euler angles 3 times one by one
they should get the same results
'''
def euler_one_by_one(euler_d_input, rot_seq):
  print('~~~~~~~~~~euler one by one~~~~~~~~~~')

  euler_r_input = euler_d_input * common.D2R
  print('euler input by %s sequence intrinsicly:' % rot_seq)
  print('%s (%s)' % (euler_d_input, euler_r_input))

  rot_once = Rotation.from_euler(rot_seq.upper(), euler_r_input)
  euler_d_once = rot_once.as_euler(rot_seq.upper(), True)
  euler_r_once = euler_d_once * common.D2R

  rot1 = Rotation.from_euler(rot_seq.upper(), [euler_r_input[0], 0, 0])
  rot2 = Rotation.from_euler(rot_seq.upper(), [0, euler_r_input[1], 0])
  rot3 = Rotation.from_euler(rot_seq.upper(), [0, 0, euler_r_input[2]])
  rot_one_by_one = rot1 * rot2 * rot3
  euler_d_one_by_one = rot_one_by_one.as_euler(rot_seq.upper(), True)
  euler_r_one_by_one = euler_d_one_by_one * common.D2R

  result = common.get_result(np.allclose(euler_d_once, euler_d_one_by_one))
  print('***euler between once and one by one: %s***' % result)
  print('euler once:\n%s (%s)' % (euler_d_once, euler_r_once))
  print('euler one by one:\n%s (%s)\n' % (euler_d_one_by_one, euler_r_one_by_one))

def test_euler_2_x():
  print('============================test euler2x============================')

  euler_d_input = np.array([20, 1, 5])
  for seq in common.ROTATION_SEQUENCES:
    euler_2_x(euler_d_input, seq)
    break # remove it to test all rotation sequences

def test_euler_one_by_one():
  print('============================test euler one by one============================')

  euler_d_input = np.array([20, 10, 15])
  for seq in common.ROTATION_SEQUENCES:
    euler_one_by_one(euler_d_input, seq)
    break # remove it to test all rotation sequences

def test():
  test_euler_2_x()
  test_euler_one_by_one()
