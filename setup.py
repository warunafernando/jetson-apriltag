#!/usr/bin/env python3
"""Setup script for jetson-apriltag package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jetson-apriltag",
    version="0.1.0",
    author="FRC Team 9202",
    description="Subprocess-based wrapper for Team 971/766 CUDA AprilTag detector on Jetson",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FRC9202/jetson-apriltag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ntcore",
    ],
    scripts=[
        "scripts/run_nt_publisher.py",
    ],
)
