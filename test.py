import numpy as np

from tests import test_util
from tests import test_euler
from tests import test_attitude
from tests import test_rotate_vectors

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  test_euler.test()
  test_attitude.test()
  test_rotate_vectors.test()

  print('\n%s: %s\n%s: %s' % (test_util.PASSED_STR, test_util.PASSED_COUNT, test_util.FAILED_STR, test_util.FAILED_COUNT))
