from setuptools import setup, find_packages

setup(
    name="ZeD_robot_evolution",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "mujoco",
        "stable-baselines3",
        "numpy",
        "pymoo",
        "matplotlib",
        "gymnasium",
    ],
)