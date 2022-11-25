# default to setuptools so that 'setup.py develop' is available,
# but fall back to standard modules that do the same

from setuptools import setup, find_packages

setup(
    name="ax_pipeline_api",
    version="0.1.1",
    description="A Library For m3axpi",
    author="Juwan",
    author_email="junhuanchen@qq.com",
    license="MIT",
    url="https://github.com/junhuanchen/ax_pipeline_api",
    packages=find_packages(

    ),
    install_requires=[
        
    ],
    tests_requires=["pytest", "scripttest"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
