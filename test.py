import numpy as np

from helpers import util

from tests import util as test_util
from tests import test_RotEx
from tests import test_euler
from tests import test_attitude
from tests import test_rotate_vectors
from tests import test_slerp
from tests import test_spline

logger = util.get_logger()
# log level is 'INFO' by default, set it as ''WARNING' if you want to close most of them
#logger.setLevel('WARNING')

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  test_RotEx.test()
  test_euler.test()
  test_attitude.test()
  test_rotate_vectors.test()

  #test_spline.test()
  #test_slerp.test()

  print('\n%s: %s\n%s: %s' % (test_util.PASSED_STR, test_util.PASSED_COUNT, test_util.FAILED_STR, test_util.FAILED_COUNT))
