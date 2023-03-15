"""Setup file for the SoulsGym module."""
from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent
long_description = (root / "README.md").read_text()

setup(
    name="soulsgym",
    packages=find_packages(),
    version="0.1.0",
    description="OpenAI gym extension for DarkSouls III",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=["Martin Schuck", "Raphael Krauthann"],
    author_email="real.amacati@gmail.com",
    url="https://github.com/amacati/SoulsGym",
    license="MIT",
    license_files=["LICENSE"],
    keywords=["gym", "reinforcement learning", "dark souls"],
    classifiers=[
        "Development Status :: 3 - Alpha", "Intended Audience :: Developers",
        "Intended Audience :: Education", "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3.10"
    ],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "gymnasium >= 0.27.0",
        "mss >= 7.0.0",
        "numpy >= 1.21.0",
        "psutil >= 5.9.0",
        "Pymem >= 1.10.0",
        "pywin32 >= 305.0",
        "PyYAML >= 6.0",
    ],
    tests_require=["pytest"],
    project_urls={
        "Homepage": "https://github.com/amacati/soulsgym",
        "Documentation": "https://soulsgym.readthedocs.io/en/latest/"
    },
)
