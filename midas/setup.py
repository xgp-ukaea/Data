from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="midas",
    version="0.3.0",
    author="Chris Bowman",
    author_email="chris.bowman.physics@gmail.com",
    description="A Multi-Instrument Divertor Analysis System (MIDAS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
