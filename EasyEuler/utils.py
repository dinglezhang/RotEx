import math
import numpy as np
import logging

ROTATION_SEQUENCES_INTRINSIC = [\
  # three axis rotation
  'ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY', \
  # two axis rotation
  'ZYZ', 'ZXZ', 'YXY', 'YZY', 'XYX', 'XZX'\
]
ROTATION_SEQUENCES_EXTRINSIC = [element.lower() for element in ROTATION_SEQUENCES_INTRINSIC]

logger = logging.getLogger()
logger.setLevel('INFO')
#logger.setLevel('WARNING')

handler = logging.StreamHandler()
fmtr = logging.Formatter(fmt="[%(asctime)s][%(filename)s:%(lineno)d][%(funcName)s()][%(levelname)s]: %(message)s")
handler.setFormatter(fmtr)
logger.addHandler(handler)

def get_logger():
  return logger

def get_angular_unit(is_degree):
  unit = 'rad'
  if is_degree:
    unit = 'deg'

  return unit

'''
Calculate the angle of a vector against XY plane.

Args:
  v: a vector
  is_degree: True is degree and False is radian for output angle
Return:
  the angle of the vector against XY plane
'''
def calc_angle_of_vector_against_XY_plane(v, is_degree = False):
  angle = math.atan2(v[2], math.sqrt(v[0] ** 2 + v[1] ** 2))
  if is_degree:
    angle = np.rad2deg(angle)
  return angle
