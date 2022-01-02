import setuptools

setuptools.setup(
    name="starter",
    version="0.0.1",
    description="Starter code.",
    packages=(setuptools.find_packages(include=["main.py", "starter", "starter.*"], exclude=["tests"])),
    author="Student",
)
