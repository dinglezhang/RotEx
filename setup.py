from codecs import open
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "RotEx",
    version = "0.2.0",
    author = "dingle",
    author_email = "zhangdingh_2004@hotmail.com",
    description = "RotEx is a set of python helper functions to apply 3D rotation, especially Euler Angles, based on scipy.spatial.transform.Rotation",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/dinglezhang/RotEx",

    include_package_data = True,
    install_requires = ['scipy'],
    extras_require = {"dev": ['pytest']},
    packages = find_packages(exclude = ["tests*"]),

    keywords = ["euler-angles", "attitude", "rotation", "3d-rotation", "rotation-extension", "rotation-help"],
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)
