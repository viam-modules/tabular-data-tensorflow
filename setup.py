from setuptools import find_packages, setup

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "keras==2.11.0",
        "keras-cv==0.5.0",
        "Keras-Preprocessing==1.1.2",
        "tflite-support",
        "viam-sdk==0.25.1",
    ],
    include_package_data=True,
)
