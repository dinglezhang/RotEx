import logging
import math

D2R = math.pi/180.0
R2D = 180.0/math.pi

ROTATION_SEQUENCES = (\
  # three axis rotation
  'zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy', \
  # two axis rotation
  'zyz', 'zxz', 'yxy', 'yzy', 'xyx', 'xzx'\
)

logger = logging.getLogger()
logger.setLevel('INFO')
#logger.setLevel('WARNING')

handler = logging.StreamHandler()
fmtr = logging.Formatter(fmt="[%(asctime)s][%(filename)s:%(lineno)d][%(funcName)s()][%(levelname)s]: %(message)s")
handler.setFormatter(fmtr)
logger.addHandler(handler)

def get_logger():
  return logger

def get_angle_unit(is_degree):
  unit = 'deg'
  if not is_degree:
    unit = 'rad'

  return unit
