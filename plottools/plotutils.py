from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from cycler import cycler
from matplotlib import (cm, colors as mplcolors, pyplot as plt,
                        rcParams)
from numpy import linspace

import sys
if sys.version_info[0] == 3:
    basestring = str

from . import colormaps


def colorscale(array=None, vmin=0, vmax=1, n=0, cmap='viridis'):
    """
    Returns a set of colors and the associated colorscale, that can be
    passed to `pylab.colorbar()`

    Optional parameters
    -------------------
        array   : array-like of floats, shape (N,)
                  values to which colors will be assigned
        vmin    : float, 0 <= vmin < vmax
                  minimum value for the color scale, on a scale from 0
                  to 1.
        vmax    : float, vmin < vmax <= 1
                  maximum value for the color scale, on a scale from 0
                  to 1.
        n       : int
                  number N of samples to draw in the range [vmin,vmax].
                  Ignored if `array` is defined.
        cmap    : str or `matplotlib.colors.ListedColormap` instance
                  colormap to be used (or its name). New colormaps
                  (viridis, inferno, plasma, magma) van be used with
                  matplotlib<2.0 using the `colormaps` module included
                  in this repository; in those cases the names must be
                  given.

    Returns
    -------
        ** If neither `array` nor `n` are defined **
        colormap : `matplotlib.colors.ListedColormap` instance
                  colormap, normalized to `vmin` and `vmax`.

        ** If either `array` or `n` is defined **
        colors  : array-like, shape (4,N)
                  array of RGBA colors
        colormap : `matplotlib.colors.ListedColormap` instance
                  colormap, normalized to `vmin` and `vmax`.

    """
    if isinstance(cmap, basestring):
        try:
            cmap = getattr(cm, cmap)
        except AttributeError:
            cmap = getattr(colormaps, cmap)
    elif type(cmap) != mplcolors.ListedColormap:
        msg = 'argument cmap must be a string or' \
              ' a matplotlib.colors.ListedColormap instance'
        raise TypeError(msg)
    # define normalization for colomap
    cnorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = cm.ScalarMappable(norm=cnorm, cmap=cmap)
    if array is None and n == 0:
        return colorbar
    # this is necessary for the colorbar to be interpreted by
    # pylab.colorbar()
    colorbar._A = []
    # now get the colors
    if array is None:
        array = linspace(vmin, vmax, n)
    colors = colorbar.to_rgba(array)
    return colors, colorbar


def savefig(output, fig=None, close=True, verbose=True, name='',
            tight=True, tight_kwargs={'pad': 0.4}):
    """
    Wrapper to save figures

    Parameters
    ----------
        output  : str
                  Output file name (including extension)

    Optional parameters
    -------------------
        fig     : pyplot.figure object
                  figure containing the plot.
        close   : bool
                  Whether to close the figure after saving.
        verbose : bool
                  Whether to print the output filename on screen
        name    : str
                  A name to identify the plot in the stdout message.
                  The message is always "Saved {name} to {output}".
        tight   : bool
                  Whether to call `tight_layout()`
        tight_kwargs : dict
                  keyword arguments to be passed to `tight_layout()`

    """
    if fig is None:
        fig = plt
    if tight:
        fig.tight_layout(**tight_kwargs)
    fig.savefig(output)
    if close:
        plt.close()
    if verbose:
        print('Saved {1} to {0}'.format(output, name))
    return


def update_rcParams(dict={}):
    """
    Update matplotlib's rcParams with any desired values. By default,
    this function sets lots of parameters to my personal preferences,
    which basically involve larger font and thicker axes and ticks,
    plus some tex configurations.

    Returns the rcParams object.

    """
    default = {}
    for tick in ('xtick', 'ytick'):
        default['{0}.major.size'.format(tick)] = 8
        default['{0}.minor.size'.format(tick)] = 4
        default['{0}.major.width'.format(tick)] = 2
        default['{0}.minor.width'.format(tick)] = 2
        default['{0}.labelsize'.format(tick)] = 20
        default['{0}.direction'.format(tick)] = 'in'
    default['xtick.top'] = True
    default['ytick.right'] = True
    default['axes.linewidth'] = 2
    default['axes.labelsize'] = 22
    default['font.family'] = 'sans-serif'
    default['font.size'] = 22
    default['legend.fontsize'] = 18
    default['lines.linewidth'] = 2
    default['text.latex.preamble']=['\\usepackage{amsmath}']
    # the matplotlib 2.x color cycle, for older versions
    default['axes.prop_cycle'] = \
        cycler(color=('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'))
    for key in default:
        # some parameters are not valid in different matplotlib functions
        try:
            rcParams[key] = default[key]
        except KeyError:
            pass
    # if any parameters are specified, overwrite anything previously
    # defined
    for key in dict:
        try:
            rcParams[key] = dict[key]
        except KeyError:
            pass
    return

