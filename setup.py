import setuptools

with open(file="./README.md", mode="r") as readme:
    long_descrition = readme.read()

setuptools.setup(
    name="agymc",
    version="0.1.dev0",
    author="MutatedFlood",
    author_email="b06901038@ntu.edu.tw",
    license="MIT",
    url="https://github.com/MutatedFlood/agymc",
    install_requires=("gym",),
    description="A concurrent wrapper for OpenAI Gym library that runs multiple environments concurrently.",
    long_description=long_descrition,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    classifiers=("Programming Language :: Python :: 3.7",),
)
