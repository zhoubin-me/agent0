import sys

from setuptools import setup, find_packages

if not sys.version.startswith('3'):
    raise Exception('Only Python 3.x is supported')

setup(name='agent0',
      packages=[package for package in find_packages()
                if package.startswith('agent0')],
      description="Ray & PyTorch based light-weight Distributed Fast Reinforcement Learning Framework",
      install_requires=[
          "gymnasium[atari]==0.28.1",
          "prefetch_generator==1.0.1",
          "dataclasses==0.6",
          "GitPython==3.1.7",
          "lz4==3.1.0",
          'pybullet==2.7.1',
      ],
      author="Zhou Bin",
      url='https://github.com/zhoubin-me/agent0',
      author_email="zhoubin.me@gmail.com",
      license='LICENSE.txt',
      version="0.51")
