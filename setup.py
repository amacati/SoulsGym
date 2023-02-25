"""Setup file for the SoulsGym module."""
from setuptools import setup, find_packages

setup(
    name="soulsgym",
    packages=find_packages(),
    version="0.1",
    description="OpenAI gym extension for DarkSouls III",
    author=["Martin Schuck", "Raphael Krauthann"],
    author_email="real.amacati@gmail.com",
    url="https://github.com/amacati/SoulsGym",
    # TODO: FILL IN CORRECT ARCHIVE
    download_url="https://github.com/amacati/SoulsGym/archive/v_01.tar.gz",
    keywords=["Reinforcement Learning", "gym", "Dark Souls"],
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha", "Intended Audience :: Developers",
        "Intended Audience :: Education", "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3.10"
    ])
