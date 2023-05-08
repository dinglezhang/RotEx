import numpy as np
from scipy.spatial.transform import Rotation

from . import common

# the key is the rotvec in space has no change. Just to get rotvec in the new frame
def euler_in_new_frame(euler_d_old_frame, euler_frame_old_2_new, rot_seq):
  print('~~~~~~~~~~euler in new frame~~~~~~~~~~')

  rot_frame_old_2_new = Rotation.from_euler(rot_seq, euler_frame_old_2_new, True)
  euler_frame_old_2_new = rot_frame_old_2_new.as_euler(rot_seq, True) # to get normalized eulers
  print('***frame from old to new:***')
  print('euler: %s\n' % euler_frame_old_2_new)

  rot_old_frame = Rotation.from_euler(rot_seq, euler_d_old_frame, True)
  euler_d_old_frame = rot_old_frame.as_euler(rot_seq, True) # to get normalized eulers
  rotvec_old_frame = rot_old_frame.as_rotvec()
  print('***rotation in old frame***:')
  print('euler:  %s' % euler_d_old_frame)
  print('rotvec: %s\n' % rotvec_old_frame)

  rotvec_new_frame = rot_frame_old_2_new.inv().apply(rotvec_old_frame)
  rot_new_frame = Rotation.from_rotvec(rotvec_new_frame)
  euler_d_new_frame = rot_new_frame.as_euler(rot_seq, True)
  print('***rotation in new frame***:')
  print('euler:  %s' % euler_d_new_frame)
  print('rotvec: %s\n' % rotvec_new_frame)

  return euler_d_new_frame
