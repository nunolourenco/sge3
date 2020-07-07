import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sge",
    version="0.0.1",
    author="Nuno Lourenço",
    author_email="naml@dei.uc.pt",
    description="Python implementation of the SGE algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nunolourenco/sge3",
    packages=['sge'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)