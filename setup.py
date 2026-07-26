from setuptools import setup, find_packages

setup(
    name="loom-video",
    version="0.1.0",
    description="Non-parametric video synthesis engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.9.0",
        "scipy>=1.11.0",
        "lz4>=4.3.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "loom=inference:main",
        ],
    },
)
