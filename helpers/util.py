import logging
import math

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
