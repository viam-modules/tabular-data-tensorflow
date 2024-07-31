from setuptools import find_packages, setup

setup(
    name="model",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "keras==2.11.0",
        "Keras-Preprocessing==1.1.2",
        "viam-sdk==0.25.1",
        "protobuf==3.20.*",
    ],
    include_package_data=True,
)
