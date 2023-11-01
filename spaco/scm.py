import matplotlib


def pal(palette_name:str, n:int) -> List[str]:
    """
    API to retrieve palettes hex code from Scientific Colour Map (SCM) by Crameri et al.'s work.

    Reference:
    Crameri, F. (2018). Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862

    Args:
        palette_name (str): name of a palette in SCM.
        n_colors (int): number of colors to retrive from the SCM palette.

    Returns:
        List[str]: a list of hex values for the retrived palette.
    """
    
    try:
        from cmcrameri import cm
    except ImportError:
        print("Please install the brilliant cmcrameri package to use Scientific colour maps.")

    return [matplotlib.colors.to_hex(i) for i in eval("cm."+palette_name+".colors")][:n]