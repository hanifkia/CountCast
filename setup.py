from setuptools import find_packages, setup


setup(
    name="countcast",
    version="0.1.0",
    description="A package for forecasting integer counts in stationary time series using MSTL, Prophet, and SARIMA.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Hanif Kia",
    author_email="kia.hanif@gmail.com",
    url="https://github.com/hanifkia/countcast",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.12.0",
        "prophet>=1.1.0",
        "pmdarima>=2.0.0",
        "pytest",
    ],
    include_package_data=True,
    package_data={
        "countcast": ["data/sample_data.csv"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
