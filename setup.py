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

from setuptools import setup, find_packages

from ax.pipeline import version
print('pipeline version: ', version)

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
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    package_data = {
        '': ['*.so'],
    },
)
