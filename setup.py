import sys

from setuptools import setup, find_packages

if not sys.version.startswith('3.10'):
    raise Exception('Only Python 3.10 is supported')

setup(name='agent0',
      packages=[package for package in find_packages()
                if package.startswith('agent0')],
      description="PyTorch based light-weight async fast Reinforcement Learning Framework",
      author="Zhou Bin",
      url='https://github.com/zhoubin-me/agent0',
      author_email="zhoubin.me@gmail.com",
      license='LICENSE.txt',
      version="0.6.0")
