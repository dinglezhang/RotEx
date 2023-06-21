import pytest
import numpy as np

@pytest.fixture()
def vector_samples():
  return np.array([
    [1, 0, 0],
    [0, -2, 0],
    [0, 0, 3],
    [0, 1, -2],
    [1, 0, 2],
    [1, -2, 0],
    [3, 4, 5],
    [3, -4, 5],
    [-3, 4, -5]
  ])
