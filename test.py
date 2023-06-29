import pytest
import numpy as np

from RotEx import utils

logger = utils.get_logger()
# log level is 'INFO' by default, set it as ''WARNING' if you want to close most of them
#logger.setLevel('WARNING')

if __name__ == '__main__':
  np.set_printoptions(precision = 8, suppress = True)

  pytest.main()
  #pytest.main(['-s', '-v'])
  #pytest.main(['-s', '-v', 'tests/test_attitude.py'])
  #pytest.main(['-s', '-v', 'tests/test_attitude.py::test_from_heading_in_enu_frame'])
