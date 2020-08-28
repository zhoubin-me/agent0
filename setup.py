import sys

from setuptools import setup, find_packages

print('Please install OpenAI Baselines (commit 8e56dd) and requirement.txt')
if not (sys.version.startswith('3.5') or sys.version.startswith('3.6')):
    raise Exception('Only Python 3.5 and 3.6 are supported')

setup(name='agent0',
      packages=[package for package in find_packages()
                if package.startswith('src')],
      install_requires=[],
      description="Ray & PyTorch based light-weight Distributed Fast Reinforcement Learning Framework",
      author="Zhou Bin",
      url='https://github.com/zmonoid/AgentZero',
      author_email="zmonoid@gmail.com",
      version="0.5")
