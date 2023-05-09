import numpy as np

from helpers import common

from tests import test_attitude as att
from tests import test_euler_2_x as e2x
from tests import test_vectors_rotation as vr

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  e2x.test()
  vr.test()
  att.test()

  print('\n%s: %s\n%s: %s' % (common.PASSED_STR, common.PASSED_COUNT, common.FAILED_STR, common.FAILED_COUNT))