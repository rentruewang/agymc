import setuptools

with open(file="./README.md", mode="r") as readme:
    long_descrition = readme.read()

setuptools.setup(
    name="agymc",
    version="0.1",
    author="MutatedFlood",
    author_email="b06901038@g.ntu.edu.tw",
    url="https://github.com/MutatedFlood/agymc",
    install_requires=("gym",),
    description="An OpenAI wrapper for concurrent batch support",
    long_descrition=long_descrition,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
