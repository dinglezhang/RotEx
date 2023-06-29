# RotEx

**RotEx** is a set of python helper functions to apply 3D rotation, like Euler Angles etc, based on _scipy.spatial.transform.Rotation_.
You can learn, understand and test Rotation quickly, and use these functions in your codes easily.

It includes:
- **rotex**, the core of module, as an extension of _scipy.spatial.transform.Rotation_. It provides basic algrithm from mathematics perspective.
- **attitude, rotate_vectors** ..., providing some functions for all kinds of applications of rotation from application perspective.
- **tests**, pytest functions for the above.
- **tools**, some tools.

**Make Euler Angles easy, not dizzy!**

# Dependencies

- scipy
- pytest

# Get started

Run test.py in the root. It calls tests on all functions provided in RotEx folder.
