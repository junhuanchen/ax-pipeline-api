##
# The MIT License (MIT)
#
# Copyright (c) 2022 junhuanchen（sipeed)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

from ax.pipeline import version
print('pipeline version: ', version)


ext_modules = [
    Pybind11Extension("m3axpi",

            include_dirs=[
                '/opt/include', '/usr/include/opencv4/', '/usr/local/include/opencv4/',
            ],
            sources= ["src/m3axpi.cpp"],
            libraries=[
                "opencv_videoio", "opencv_highgui", "opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_freetype", "opencv_freetype", "sample_vin_ivps_npu_vo_sipy"
            ],
            extra_compile_args=['-std=c++11', '-std=gnu++11'],
            extra_link_args=[
                "-Lax/lib",
                "-L/usr/local/lib/python3.9/dist-packages/ax/lib/",
                "-Wl,-rpath=/usr/local/lib/python3.9/dist-packages/ax/lib/",
                "-Wno-format-security",
            ],
        ),
]

setup(
    name="ax-pipeline-api",
    version=version,
    license='MIT',
    description="A Python API For wiki.sipeed.com/m3axpi Pipeline",
    author="Juwan",
    author_email="junhuanchen@qq.com",
    long_description=open('README.md', 'r', encoding='UTF-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/junhuanchen/ax-pipeline-api",
    packages=[
        "ax",
    ],
    install_requires=[

    ],
    ext_modules=ext_modules,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    package_data = {
        '': ['*.so'],
    },
)
