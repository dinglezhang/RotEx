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
