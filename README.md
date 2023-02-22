# Spaco: a comprehensive tool for coloring spatial data at single-cell resolution

Visualizing spatially resolved biological data with appropriate color mapping can significantly facilitate the exploration of underlying patterns and heterogeneity. Spaco (**spa**tial **co**lorization) provides a spatially constrained approach that generates discriminate color assignments for visualizing single-cell spatial data in various scenarios.

[Quick Example](https://github.com/Bai-Lab/Spaco2/blob/main/notebooks/demo.ipynb) - [Citation](https://github.com/Bai-Lab/Spaco2)

# Installation

```
pip install git+https://github.com/Bai-Lab/Spaco2.git
```

# Usage (under development)

Spaco tutorials is still under development, see our [Quick Example](https://github.com/Bai-Lab/Spaco2/blob/main/notebooks/demo.ipynb).

# Development Process
## Code quality
- File and function docstrings should be written in [Google style](https://google.github.io/styleguide/pyguide.html)
- We use `black` to automatically format code in a standardized format. To ensure that any code changes are up to standard, use `pre-commit` as such.
```
# Run the following two lines ONCE.
pip install pre-commit
pre-commit install
```
Then, all future commits will call `black` automatically to format the code. Any code that does not follow the standard will cause a check to fail.
