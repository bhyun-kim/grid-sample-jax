from setuptools import setup, find_packages

setup(
  name="grid-sample-jax",
  version="0.1.0",
  packages=find_packages(),
  install_requires=[
    # "torch", Install manually to avoid conflicts to existing torch 
    # "jax", Install manually to avoid conflicts to existing jax 
    "requests",
    "Pillow",
    "matplotlib",
    "psutil",
  ],
  python_requires=">=3.7",
  author="bhyun-kim",
  description="JAX implementation of torch.nn.functional.grid_sample",
  url="https://github.com/bhyun-kim/grid-sample-jax",
)
