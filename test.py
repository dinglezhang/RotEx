import numpy as np

from tests import test_util
from tests import test_euler_2_x
from tests import test_rotate_vectors
from tests import test_attitude

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  test_euler_2_x.test()
  test_rotate_vectors.test()
  test_attitude.test()

  print('\n%s: %s\n%s: %s' % (test_util.PASSED_STR, test_util.PASSED_COUNT, test_util.FAILED_STR, test_util.FAILED_COUNT))
