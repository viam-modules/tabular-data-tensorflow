from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "keras==2.14.0",
        "Keras-Preprocessing",
        "numpy<2",
        "tensorflow==2.14.0",
        "viam-sdk",
    ],
    include_package_data=True,
)
