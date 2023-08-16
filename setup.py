from setuptools import setup


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()


setup(
    name="spaco",
    version="0.1.0",
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    packages=["spaco"],
    author="Zehua Jing, Qianhua Zhu, Hailin Pan",
    author_email="jingzehua@genomics.cn",
    description="Spaco: a comprehensive tool for coloring spatial data at single-cell resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL",
    url="https://github.com/BrainStOrmics/Spaco",
)
