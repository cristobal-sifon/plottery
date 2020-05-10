
"""Tools for common astronomical plots

Requires `astLib` and `astropy`
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astLib import astCoords
from astLib.astWCS import WCS
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage


def contour_overlay(
        ax, imgfile, contourfile, smoothing='gaussian_filter', args=(3,),
        smoothing_kwargs={}, **contour_kwargs):
    """Overlay contours from an external image

    Parameters
    ----------
    ax : `matplotlib.axes.Axes` instance
        axis where the image is plotted, initialized e.g., through
        `plt.axes` or `plt.subplot`
    imgfile : `str`
        filename of the FITS image to be shown, or the FITS image
        associated with the color image to be shown
    contourfile : `str`
        filename of the image from which to overlay contours
    smoothing : `scipy.ndimage` attribute
        name of any appropriate `scipy.ndimage` function. Set to `None`
        to disable.
    args : `tuple`
        positional arguments to pass to the function defined by
        `smoothing` (e.g., sigma for `gaussian_filter`)
    smoothing_kwargs : `dict` (optional)
        keyword arguments passed to the function defined by `smoothing`
    contour_kwargs : `dict`
        `plt.contour` keyword arguments (e.g., levels, colors)
    """
    # for some reason astWCS can read some files that astropy.wcs
    # cannot
    imgwcs = WCS(imgfile)
    contourwcs = WCS(contourfile)
    contourdata = np.array(fits.getdata(contourfile), dtype=float)
    while len(contourdata.shape) > 2:
        contourdata = contourdata[0]
    # convert coords
    ny, nx = contourdata.shape
    xo, yo = contourwcs.pix2wcs(-1, -1)
    x1, y1 = contourwcs.pix2wcs(nx, ny)
    # astropy - is the pixel numbering convention correct?
    #xo, yo = contourwcs.wcs_pix2world(0, 0)
    #x1, y1 = contourwcs.wcs_pix2world(nx, ny)
    xo, yo = imgwcs.wcs2pix(xo, yo)
    x1, y1 = imgwcs.wcs2pix(x1, y1)
    # astropy
    #xo, yo = imgwcs.wcs_world2pix(xo, yo)
    #x1, y1 = imgwcs.wcs_world2pix(x1, y1)
    if smoothing is not None:
        smoothing_func = getattr(ndimage, smoothing)
        contourdata = smoothing_func(contourdata, *args, **smoothing_kwargs)
    contours = ax.contour(contourdata, extent=(xo,x1,yo,y1), **contour_kwargs)
    return contours


def format_wcs(x, sep=':'):
    """
    Replace the 60's for 0's and change other values consistently,
    and add 0's at the beginning of single-digit values

    Generally not intended for stand-alone use, but rather this
    function is called by `wcslabels`

    Parameters
    ----------
    x : `str`
        Sky coordinate string in hms or dms format
    sep : `char`
        Character used to separate hours/degrees, minutes and seconds

    """
    x = x.split(sep)
    x[2] = '{0:02.0f}'.format(float(x[2]))
    for i in (1, 0):
        if x[i+1] == '60':
            x[i+1] = '00'
            x[i] = str(int(x[i]) + 1)
    #south = True if x[0][0] == '-' else False
    return ':'.join(x)


def phase_space(
        R, v, sigma_v=0, hist_bins=10, ylim=None, vertlines=None,
        xlabel=r'$R\,({\rm Mpc})$',
        ylabel=r'$v_{\rm gal}\,(\mathrm{km\,s^{-1}})$'):
    """
    Plot the phase space (distance vs. velocity) of galaxies. Used mostly for
    galaxy cluster membership diagnostics.

    Parameters
    ----------
        R       : array of floats
                  cluster-centric distances
        v       : array of floats
                  peculiar velocities
        sigma_v : float (optional)
                  cluster velocity dispersion
        hist_bins : int or list (optional)
                  bins or number of bins for the velocity histogram
        ylim    : tuple of floats, length 2 (optional)
                  y-axis limits
        vertlines : (list of) floats or (list of) length-2 tuples with
                            each element containing (loc, linestyle)
                  locations at which to plot vertical lines, for instance
                  to mark r200 or other characteristic radii
                  NOTE: maybe also add color and linewidth to the input later

    """
    fig = plt.figure(figsize=(7,4))
    ax = plt.subplot2grid((1,4), (0,0), colspan=3)
    ax.plot(R, v, 'k.')
    ax.axhline(0, ls='-', color='k', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xlim = ax.get_xlim()
    ax.set_xlim(-0.1, xlim[1])
    ax.axvline(0, ls='--', color='k', lw=1)
    if vertlines is not None:
        if not hasattr(vertlines, '__iter__'):
            vertlines = [vertlines]
        if hasattr(vertlines[0], '__iter__'):
            for vl in vertlines:
                ax.axvline(vl[0], ls=vl[1], color='k', lw=1)
        else:
            for vl in vertlines:
                ax.axvline(vl[0], ls=':', color='k', lw=1)
    if ylim is None:
        ylim = ax.get_ylim()
    else:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%s$'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%d$'))
    right = plt.subplot2grid((1,4), (0,3))
    n, edges, patches = right.hist(v, hist_bins, orientation='horizontal',
                                   histtype='stepfilled', color='y')
    if sigma_v > 0:
        n_area = (n * (edges[1:] - edges[:-1])).sum()
        t = np.linspace(ylim[0], ylim[1], 101)
        x = (t[1:] + t[:-1]) / 2
        f = np.exp(-x**2/(2*sigma_v**2)) / ((2*np.pi)**2*sigma_v)
        f_area = (f * (t[1:] - t[:-1])).sum()
        right.plot(f/f_area*n_area, x, '-', color=(1,0,0))
    right.xaxis.set_major_locator(ticker.MaxNLocator(3))
    right.xaxis.set_major_formatter(ticker.FormatStrFormatter('$%d$'))
    right.set_yticklabels([])
    right.set_xlabel(r'$N(v_{\rm gal})$')
    fig.tight_layout(pad=0.2)
    return fig, [ax, right]


def wcslabels(wcs, xlim, ylim, xsep='00:00:01', ysep='00:00:15',
              ax=None, label_color='k', tick_color='w', axes_color=None,
              rotate_x=0, rotate_y=90):
    """
    Get WCS ticklabels

    Parameters
    ----------
        wcs     : astWCS.WCS instance
                  the wcs of the image to be shown
        xlim, ylim : sequences of length 2
                  the minimum and maximum values of the x and y axes,
                  in pixels
        xsep    : string
                  separation of right ascension ticks in the x axis,
                  in colon-separated hms format
        xsep    : string
                  separation of declination ticks in the y axis, in
                  colon-separated dms format
        ax      : matplotlib.Axes instance (optional)
                  if provided, the ticks will be displayed on it
        label_color : string or matplotlib color
                  color with which the tick labels will be displayed,
                  if ax is provided
        tick_color : string or matplotlib color
                  color for the ticks, applied to both axes.
        axes_color : string or matplotlib color (optional)
                  color for the axes. If not set, `tick_color` will be
                  used.
        rotate_x : float
                  xticklabel rotation, in deg
        rotate_y : float
                  yticklabel rotation, in deg

    Returns
    -------
        [xticks, xticklabels] : lists containing the positions and
                  labels for right ascension hms labels
        [yticks, yticklabels] : lists containing the positions and
                  labels for declination dms labels

    Note : this function assumes that RA runs along the x axis and Decl
        runs along the y axis. Will generalize in the future.

    """
    left, right = xlim
    bottom, top = ylim
    wcslim = [wcs.pix2wcs(left, bottom), wcs.pix2wcs(right, top)]
    ralim, declim = np.transpose(wcslim)
    rasep = astCoords.hms2decimal(xsep, ':')
    decsep = astCoords.dms2decimal(ysep, ':')
    raticks = np.arange(0, max(ralim), rasep)
    raticks = raticks[raticks > min(ralim)]
    decticks = np.arange(-90, max(declim), decsep)
    decticks = decticks[decticks > min(declim)]
    # this assumes that the rotation angle of the image is 0/90/180/270
    # degrees
    xticks = [wcs.wcs2pix(x, round(declim[0], 4))[0] for x in raticks]
    yticks = [wcs.wcs2pix(round(ralim[0], 4), y)[1] for y in decticks]
    xticklabels = [astCoords.decimal2hms(t, ':') for t in raticks]
    yticklabels = [astCoords.decimal2dms(t, ':') for t in decticks]
    # this corrects for machine-precision errors, which can be on the
    # order of a couple hundredths of a second
    if np.array(xsep.split(':')[:2], dtype=int).sum() > 0:
        xticklabels = [':'.join([t.split(':')[0], t.split(':')[1], '00'])
                       for t in xticklabels]
    if np.array(ysep.split(':')[:2], dtype=int).sum() > 0:
        yticklabels = [':'.join([t.split(':')[0], t.split(':')[1], '00'])
                       for t in yticklabels]
    # format properly (remove 60's and add 0's)
    xticklabels = np.array([format_wcs(xt) for xt in xticklabels])
    yticklabels = np.array([format_wcs(yt) for yt in yticklabels])
    # get tick positions for rounded labels
    raticks = [astCoords.hms2decimal(xt, ':') for xt in xticklabels]
    decticks = [astCoords.dms2decimal(yt, ':') for yt in yticklabels]
    xticks = np.array([wcs.wcs2pix(x, declim[0])[0] for x in raticks])
    yticks = np.array([wcs.wcs2pix(ralim[0], y)[1] for y in decticks])
    # it seems they must be sorted?
    j = np.argsort(xticks)
    xticks = xticks[j]
    xticklabels = xticklabels[j]
    j = np.argsort(yticks)
    yticks = yticks[j]
    yticklabels = yticklabels[j]
    # display?
    if ax:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        plt.setp(ax.xaxis.get_ticklabels(), rotation=rotate_x)
        if rotate_y == 90:
            va = 'center'
        else:
            va = 'top'
        plt.setp(ax.yaxis.get_ticklabels(), rotation=rotate_y, va=va)
        # set tick colors. Doing it here so I don't need to remember
        # these three commands every time, plus the order seems to
        # matter here
        if axes_color is None:
            axes_color = tick_color
        for key, spine in ax.spines.items():
            spine.set_color(axes_color)
        ax.tick_params(axis='both', which='both', colors=tick_color)
        ax.set_xticklabels(xticklabels, color=label_color)
        ax.set_yticklabels(yticklabels, color=label_color)
    return [xticks, xticklabels], [yticks, yticklabels]



