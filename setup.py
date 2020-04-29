from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpt2-poet",
    version="0.0.1",
    author="kianfucius",
    author_email="kian.salamzadeh@hotmail.com",
    description="A small package for testing gpt-2 finetuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kianfucius/gpt2-poet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['transformers','torch', 'numpy']
)
