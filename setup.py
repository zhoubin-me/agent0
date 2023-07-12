import sys

from setuptools import setup, find_packages

if not sys.version.startswith('3.10'):
    raise Exception('Only Python 3.10 is supported')

setup(name='agent0',
      packages=[package for package in find_packages()
                if package.startswith('agent0')],
      description="PyTorch based light-weight async fast Reinforcement Learning Framework",
      author="Zhou Bin",
      install_requires=[
          "gymnasium[atari,accept-rom-license,other]==0.28.1",
          "prefetch_generator==1.0.1",
          "lz4==3.1.0",
          "tensorboardX==2.6.1",
          "tqdm==4.65.0",
          "hydra-core==1.3.2",
          "dacite==1.8.1",
          "einops==0.6.1"
      ],
      url='https://github.com/zhoubin-me/agent0',
      author_email="zhoubin.me@gmail.com",
      license='LICENSE.txt',
      version="0.6.0")
