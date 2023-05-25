from scipy.spatial.transform import Rotation

from . import util

logger = util.get_logger()

'''
Get rotation in new frame from old frame.
The key algorism is the rotvec in space has no change. Just to get rotvec in the new frame.

Args:
  rot_old_frame: body rotation in old frame
  rot_frame_old_2_new: frame rotation from old frame to new
Return:
  body rotation in new frame
'''
def rot_in_new_frame(rot_old_frame, rot_frame_old_2_new):
  rotvec_old_frame = rot_old_frame.as_rotvec()
  logger.info('body rotation in old frame: rotvec%s' % rotvec_old_frame)

  rotvec_new_frame = rot_frame_old_2_new.inv().apply(rotvec_old_frame)
  rot_new_frame = Rotation.from_rotvec(rotvec_new_frame)
  logger.info('body rotation in new frame: rotvec%s' % rotvec_new_frame)

  return rot_new_frame

'''
Get euler angles in new frame from old frame.

Args:
  euler_old_frame: body euler angles in old frame
  euler_frame_old_2_new: frame euler angles from old frame to new
  rot_seq: euler angles rotation sequence
  is_degree: True is degree and False is radian for both input and output eulers
Return:
  body euler angles in new frame
'''
def euler_in_new_frame(euler_old_frame, euler_frame_old_2_new, rot_seq, is_degree):
  rot_old_frame = Rotation.from_euler(rot_seq, euler_old_frame, is_degree)
  logger.info('body rotation in old frame: euler(%s)%s' % (util.get_angle_unit(is_degree), rot_old_frame.as_euler(rot_seq, is_degree))) # call as_euler() to get normalized euler

  rot_frame_old_2_new = Rotation.from_euler(rot_seq, euler_frame_old_2_new, is_degree)
  logger.info('frame rotation from old to new: euler(%s)%s' % (util.get_angle_unit(is_degree), rot_frame_old_2_new.as_euler(rot_seq, is_degree)))

  rot_new_frame = rot_in_new_frame(rot_old_frame, rot_frame_old_2_new)
  euler_new_frame = rot_new_frame.as_euler(rot_seq, is_degree)
  logger.info('body rotation in new frame: euler(%s)%s' % (util.get_angle_unit(is_degree), euler_new_frame))

  return euler_new_frame
