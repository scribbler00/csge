from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="dies",
    version="0.1.0a0",
    description="Coopetitive Soft Gating Ensemble.",
    long_description=readme(),
    url="https://github.com/scribbler00/csge",
    keywords="machine learning, ensemble, ensemble members",
    author="Maarten Bieeshaar",
    author_email="iescloud@uni-kassel.de",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "numpy",
        "sklearn",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "nose",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
    ],
)
