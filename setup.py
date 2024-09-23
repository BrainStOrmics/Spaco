from setuptools import setup, find_packages


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()


setup(
    name="spaco-release",
    version="0.2.2",
    python_requires=">=3.7",
    #install_requires=read_requirements("requirements.txt"),
    install_requires=[
        'anndata>=0.8.0',
        'colormath>=3.0.0',
        'numpy>=1.18.0',
        'pandas>=0.25.1',
        'pyciede2000==0.0.21',
        'scikit-image>=0.19.0',
        'scikit-learn>=0.19.0',
        'scipy>=1.10.0',
        'typing_extensions>=4.0.0',
        'umap-learn>=0.5.0',
    ],
    packages=find_packages(),
    author="Zehua Jing, Qianhua Zhu, Hailin Pan",
    author_email="jingzehua@genomics.cn",
    description="Spaco: a comprehensive tool for coloring spatial data at single-cell resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL",
    url="https://github.com/BrainStOrmics/Spaco",
)
