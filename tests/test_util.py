import numpy as np

PASSED_STR = "\033[1;32mPASSED\033[0m"   # with green color
FAILED_STR = "\033[1;31mFAILED\033[0m"   # with red color

PASSED_COUNT = 0
FAILED_COUNT = 0

def get_result(is_passed):
  global PASSED_COUNT
  global FAILED_COUNT

  if is_passed:
    PASSED_COUNT = PASSED_COUNT + 1
    return PASSED_STR
  else:
    FAILED_COUNT = FAILED_COUNT + 1
    return FAILED_STR

test_vectors = np.array([
  [1, 0, 0],
  [0, 2, 0],
  [0, 0, 3],
  [0, 1, 2],
  [1, 0, 2],
  [1, 2, 0],
  [3, 4, 5]
])

def get_test_vectors():
  return test_vectors
