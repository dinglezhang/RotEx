# EasyEuler

**EasyEuler** is a set of helper functions to apply 3D rotation, especially Euler Angles, based on _scipy.spatial.transform.Rotation_.
You can learn, understand and test Rotation quickly, and use these functions in your codes easily.

It includes:

- RotEx: 
  This is the core of EasyEuler, as an extension of _scipy.spatial.transform.Rotation_.
  It provides basic algrithm from mathematics perspective.

- attitude, rotate_vectors ...: 
  They provide some functions for all kinds of applications of rotation from application perspective.

- tests: 
  pytest functions for the above.

**Make Euler Angles easy, not dizzy!**

# Requirements

- numpy
- scipy
- pytest

# Get started

Run test.py in the root. It calls tests on all functions provided in EasyEuler folder.
