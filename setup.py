from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "keras==2.14.0",
        "Keras-Preprocessing",
        "viam-sdk==0.25.2",
        "protobuf==4.25.3",
    ],
    include_package_data=True,
)
