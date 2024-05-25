import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AIFramework",
    version="0.0.19",
    author="Pasquale Casciano",
    author_email="pa.casciano@gmail.com",
    description="A tiny AI learner framework based on Jeremy Howard's lessons.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bigghis/AIFramework",
    packages=setuptools.find_packages(),
     install_requires=[
        'torch',
        'torchvision',
        'torcheval',
        'fastcore',
        'numpy',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
