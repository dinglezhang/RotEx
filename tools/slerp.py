import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def try_slerp():
  times = [0, 10, 20, 40]
  angles = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
  rotations = Rotation.from_euler('XYZ', angles, degrees = True)

  slerp = Slerp(times, rotations)

  angles = slerp(times).as_euler('XYZ', degrees = True)

  times_plot = np.linspace(times[0], times[-1], 100)
  angles_plot = slerp(times_plot).as_euler('XYZ', degrees = True)

  import matplotlib.pyplot as plt
  plt.plot(times_plot, angles_plot, '-1')
  plt.plot(times, angles, 'x')
  plt.plot("Euler angles")
  plt.show()

if __name__ == '__main__':
  try_slerp()
