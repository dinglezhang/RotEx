from scipy.spatial.transform import Rotation

from . import utils

logger = utils.get_logger()

'''
Get vectors rotated by rotation.

Args:
  vectors: vectors to be rotated
  rot_once: rotation once
  times: times to apply the rotation
  on_frame: True is to rotate frame, False is to rotate vectors
Return:
  vectors rotated
'''
def rotate_vectors(vectors, rot_once, times = 1, on_frame = False):
  times_str = 'once'
  if times > 1:
    times_str = '%s times' % times
  on_str = 'on vector'
  if on_frame:
    on_str = 'on frame'

  logger.info('to rotate vectors %s %s' % (times_str, on_str))
  logger.info('vectors input: %s' % vectors)

  rot = rot_once
  for i in range(1, times):
    #logger.info('times: %s' % i)
    rot = rot * rot_once

  vectors_rotated = rot.apply(vectors, on_frame)
  logger.info('vectors rotated: %s' % vectors_rotated)

  return vectors_rotated

'''
Get vectors rotated by euler angles.

Args:
  vectors: vectors to be rotated
  euler: euler angles to rotate
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for input euler
  times: times to apply the rotation
  on_frame: True is to rotate frame, False is to rotate vectors
Return:
  vectors rotated
'''
def rotate_vectors_by_euler(vectors, euler, rot_seq, is_degree, times = 1, on_frame = False):
  logger.info('euler(%s) input in %s sequence: %s' % (utils.get_angular_unit(is_degree), rot_seq, euler))

  rot_once = Rotation.from_euler(rot_seq, euler, is_degree)
  vectors_rotated = rotate_vectors(vectors, rot_once, times, on_frame)

  return vectors_rotated
