import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline

def try_spline():
  times = [0, 10, 20, 40]
  angles = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
  rotations = Rotation.from_euler('XYZ', angles, degrees = True)

  spline = RotationSpline(times, rotations)

  angles = spline(times, 0).as_euler('XYZ', degrees = True)
  angular_rate = np.rad2deg(spline(times, 1))
  angular_acceleration = np.rad2deg(spline(times, 2))

  times_plot = np.linspace(times[0], times[-1], 30)
  angles_plot = spline(times_plot).as_euler('XYZ', degrees = True)
  angular_rate_plot = np.rad2deg(spline(times_plot, 1))
  angular_acceleration_plot = np.rad2deg(spline(times_plot, 2))

  import matplotlib.pyplot as plt
  plt.plot(times_plot, angles_plot, '-1')
  plt.plot(times, angles, 'x')
  plt.plot("Euler angles")
  plt.show()

  plt.plot(times_plot, angular_rate_plot, '-1')
  plt.plot(times, angular_rate, 'x')
  plt.title("Angular rate")
  plt.show()

  plt.plot(times_plot, angular_acceleration_plot, '-1')
  plt.plot(times, angular_acceleration, 'x')
  plt.title("Angular acceleration")
  plt.show()

if __name__ == '__main__':
  try_spline()
