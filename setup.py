from setuptools import setup, find_packages

setup(
    name="blipsearch",
    version="0.1.0",
    description="BLIP-based on-device natural language video search engine",
    author="Masterchief",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "torchvision",
        "opencv-python",
        "pillow",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "blipsearch=blipsearch.cli:main",
        ],
    },
    python_requires=">=3.11",
)
