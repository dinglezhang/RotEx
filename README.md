# RotEx

**RotEx** is a set of python helper functions to apply 3D rotation, like Euler Angles etc, based on _scipy.spatial.transform.Rotation_.
You can learn, understand and test Rotation quickly, and use these functions in your codes easily.

It includes:

- **rotex**, the core of module, as an extension of _scipy.spatial.transform.Rotation_. It provides basic algrithm from mathematics perspective.
- **attitude, rotate_vectors** ..., providing some functions for all kinds of applications of rotation from application perspective.
- **tests**, pytest functions for the above, for dev only.
- **tools**, some tools, for dev only.

Please see the comments above all functions in source code to get details.

Please ref to ./tests/* code to get some examples to call these functions.

**Make rotation easy, not dizzy!**

## Dependencies

- python >3.0
- scipy
- pytest

## Installation

- Install from the [Python Package Index](https://pypi.org/project/RotEx/):

  ```bash
  pip install RotEx
  ```

- Install from the source:

  ```bash
  cd RotEx
  pip install -e .
  ```

## Get started

- Use the RotEx module

  ```python
  from RotEx import utils
  from RotEx import rotex
  from RotEx import attitude
  from RotEx import rotate_vectors

  (rot, att) = attitude.from_heading_in_enu_frame(np.array([-1, 2, 3]), 15, True)
  ...
  ```

- Try test RotEx (only for from source)

  Run test.py in the root. It calls tests on all functions provided in RotEx folder.
