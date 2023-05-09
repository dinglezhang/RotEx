import logging
import math

D2R = math.pi/180.0
R2D = 180.0/math.pi

PASSED_STR = "\033[1;32mPASSED\033[0m"   # with green color
FAILED_STR = "\033[1;31mFAILED\033[0m"   # with red color

PASSED_COUNT = 0
FAILED_COUNT = 0

ROTATION_SEQUENCES = (\
  # three axis rotation
  'zyx', 'zxy', 'yxz', 'yzx', 'xyz', 'xzy', \
  # two axis rotation
  'zyz', 'zxz', 'yxy', 'yzy', 'xyx', 'xzx'\
)

logger = logging.getLogger()
logger.setLevel('INFO')

handler = logging.StreamHandler()
fmtr = logging.Formatter(fmt="[%(asctime)s][%(filename)s:%(lineno)d][%(funcName)s()][%(levelname)s]: %(message)s")
handler.setFormatter(fmtr)
logger.addHandler(handler)

def get_logger():
  return logger

def get_result(is_passed):
  global PASSED_COUNT
  global FAILED_COUNT

  if is_passed:
    PASSED_COUNT = PASSED_COUNT + 1
    return PASSED_STR
  else:
    FAILED_COUNT = FAILED_COUNT + 1
    return FAILED_STR
