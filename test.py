import numpy as np

from helpers import util

from tests import test_util
from tests import test_euler
from tests import test_attitude
from tests import test_rotate_vectors

logger = util.get_logger()
# log level is 'INFO' by default, set it as ''WARNING' if you want to close most of them
#logger.setLevel('WARNING')

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  test_euler.test()
  test_attitude.test()
  test_rotate_vectors.test()

  print('\n%s: %s\n%s: %s' % (test_util.PASSED_STR, test_util.PASSED_COUNT, test_util.FAILED_STR, test_util.FAILED_COUNT))
