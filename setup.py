import setuptools

setuptools.setup(
    name="voh",
    version="0.0.0",
    description="Voice of Heart",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="",
    author="Francis Lim",
    author_email="thyeem@gmail.com",
    license="MIT",
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="asr",
    packages=setuptools.find_packages(),
    install_requires=[],
    python_requires=">=3.6",
)
