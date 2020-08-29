import sys

from setuptools import setup, find_packages

if not (sys.version.startswith('3')):
    raise Exception('Only Python 3.x is supported')

setup(name='agent0',
      packages=[package for package in find_packages()
                if package.startswith('src')],
      install_requires=[],
      description="Ray & PyTorch based light-weight Distributed Fast Reinforcement Learning Framework",
      author="Zhou Bin",
      url='https://github.com/zmonoid/AgentZero',
      author_email="zmonoid@gmail.com",
      version="0.51")
