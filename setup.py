"""
MSP package setup file.
"""
from setuptools import find_packages, setup

with open("README.md") as f:
    long_desc = f.read()

setup(
    name="msp",
    version="0.1",
    description="Machine Scheduling Problem(MSP) library to construct an optimize schedule.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/bilalsp/msp",
    author="Mohammed Bilal Ansari",
    author_email="mohammedbilalansari.official@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "absl-py==0.13.0",
        "astunparse==1.6.3",
        "cached-property==1.5.2; python_version < '3.8'",
        "cachetools==4.2.2; python_version ~= '3.5'",
        "certifi==2021.5.30",
        "charset-normalizer==2.0.4; python_version >= '3'",
        "clang==5.0",
        "cloudpickle==1.6.0; python_version >= '3.5'",
        "decorator==5.0.9; python_version >= '3.5'",
        "dm-tree==0.1.6",
        "flatbuffers==1.12",
        "gast==0.4.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "google-auth==1.35.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'",
        "google-auth-oauthlib==0.4.5; python_version >= '3.6'",
        "google-pasta==0.2.0",
        "grpcio==1.39.0",
        "h5py==3.1.0; python_version >= '3.6'",
        "idna==3.2; python_version >= '3'",
        "importlib-metadata==4.8.0; python_version < '3.8'",
        "keras==2.6.0",
        "keras-preprocessing==1.1.2",
        "markdown==3.3.4; python_version >= '3.6'",
        "numpy==1.19.5; python_version == '3.7'",
        "oauthlib==3.1.1; python_version >= '3.6'",
        "opt-einsum==3.3.0; python_version >= '3.5'",
        "protobuf==3.17.3",
        "pyasn1==0.4.8",
        "pyasn1-modules==0.2.8",
        "pyyaml==5.4.1",
        "requests==2.26.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'",
        "requests-oauthlib==1.3.0",
        "rsa==4.7.2; python_version >= '3.6'",
        "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "tensorboard==2.6.0; python_version >= '3.6'",
        "tensorboard-data-server==0.6.1; python_version >= '3.6'",
        "tensorboard-plugin-wit==1.8.0",
        "tensorflow==2.6.0",
        "tensorflow-estimator==2.6.0",
        "tensorflow-probability==0.13.0",
        "termcolor==1.1.0",
        "typing-extensions==3.7.4.3; python_version < '3.8'",
        "urllib3==1.26.6; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
        "werkzeug==2.0.1; python_version >= '3.6'",
        "wheel==0.37.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "wrapt==1.12.1",
        "zipp==3.5.0; python_version >= '3.6'",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["operational research", "python", "scheduling", "deep learning"],
)