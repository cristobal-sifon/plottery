from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy
import pylab
from astLib import astCoords, astWCS
from astropy.io import fits
from cycler import cycler
from itertools import count
try:
    from itertools import izip
except ImportError:
    basestring = str
    izip = zip
    xrange = range
from matplotlib import cm, colors as mplcolors, rcParams, ticker
from scipy import optimize
from scipy.ndimage import zoom

# in case matplotlib.__version__ < 1.5
import colormaps


__version__ = '0.2.5'


def contour_levels(x, y=[], bins=10, levels=(0.68,0.95)):
    """
    Get the contour levels corresponding to a set of percentiles (given as
    fraction of 1) for a 2d histogram.

    Parameters
    ----------
        x : array of floats
            if y is given then x must be a 1d array. If y is not given then
            x should be a 2d array
        y : array of floats (optional)
            1d array with the same number of elements as x
        bins : argument of numpy.histogram2d
        levels : list of floats between 0 and 1
            the fractional percentiles of the data that should be above the
            returned values

    Returns
    -------
        level_values : list of floats, same length as *levels*
            The values of the histogram above which the fractional percentiles
            of the data given by *levels* are

    """
    if len(y) > 0:
        if len(x) != len(y):
            msg = 'Invalid input for arrays; must be either 1 2d array'
            msg += ' or 2 1d arrays'
            raise ValueError(msg)
    else:
        if len(numpy.array(x).shape) != 2:
            msg = 'Invalid input for arrays; must be either 1 2d array'
            msg += ' or 2 1d arrays'
            raise ValueError(msg)
    def findlevel(lo, hist, level):
        return 1.0 * hist[hist >= lo].sum()/hist.sum() - level
    if len(x) == len(y):
        hist, xedges, yedges = numpy.histogram2d(x, y, bins=bins)
        hist = numpy.transpose(hist)
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    elif len(y) == 0:
        hist = numpy.array(x)
    level_values = [optimize.bisect(findlevel, hist.min(), hist.max(),
                                    args=(hist,l)) for l in levels]
    return level_values


def contours_external(ax, imgwcs, contourfile, levels, colors, lw=1):
    """
    Draw contours from contourfile in the frame of imgwcs.

    """
    contourwcs = astWCS.WCS(contourfile)
    contourdata = fits.getdata(contourfile)
    while len(contourdata.shape) > 2:
        contourdata = contourdata[0]
    # convert coords
    ny, nx = contourdata.shape
    xo, yo = contourwcs.pix2wcs(-1, -1)
    x1, y1 = contourwcs.pix2wcs(nx, ny)
    xo, yo = imgwcs.wcs2pix(xo, yo)
    x1, y1 = imgwcs.wcs2pix(x1, y1)
    contourdata = zoom(contourdata, 3, order=3)
    ax.contour(contourdata, levels, colors=colors, linewidths=lw,
               extent=(xo,x1,yo,y1))
    return


def corner(X, config=None, names='', labels=None, bins=20, bins1d=20,
           clevels=(0.68,0.95), contour_reference='samples',
           truths=None, truths_in_1d=False, truth_color='r',
           smooth=False, likelihood=None, likesmooth=1,
           color_likelihood='r', colors='k', cmap=None,
           ls1d='-', ls2d='solid', style1d='curve', medians1d=True,
           percentiles1d=True, background=None, bweight=None, bcolor='r',
           alpha=0.5, limits=None, show_likelihood_1d=False,
           ticks=None, show_contour=True, top_labels=False,
           pad=1, h_pad=0.1, w_pad=0.1, output='', verbose=False,
           names_kwargs={}, **kwargs):
    """
    Do a corner plot (e.g., with the posterior parameters of an MCMC chain).
    Note that there may still be some issues with the tick labels.

    Parameters
    ----------
      X         : array-like
                  all posterior parameters. Can also be the outputs of
                  more than one chain, given as an array of arrays of models.
                  For instance, the example has three chains with two
                  parameters. In this case, X = [[A1,B1], [A2,B2], [A3,B3]].

    Optional parameters
    -------------------
      config    : str (optional - NOT YET IMPLEMENTED)
                  name of file containing any parameters whose default values
                  should be modified. Format of the file is two columns,
                  where the first is the parameter name as listed here,
                  and the second is the value for that parameter. If the
                  parameter takes a list of values they should be comma-
                  separated, and multiple entries semi-colon-separated.
                  For example, a file containing
                        bins             20
                        bins1d           50
                        colors       yellow
                        ticks     2,3,4;10,11,12;3.2,3.3,3.4
                  would only modify these parameters. Note that because of the
                  content of the 'ticks' parameter, the chain must be a
                  three-parameter model.
      names     : list of strings
                  Names for each of the chains. Will be used to show a legend
                  in the (empty) upper corner
      labels    : list of strings
                  names of the parameters
      bins      : int or array of ints
                  Number of bins for the contours in the off-diagonal panels.
                  Should be one value per chain, one value per parameter,
                  or have shape (nchains,nparams)
      bins1d    : int or array of ints
                  Number of bins for the histograms or curves in the diagonal
                  panels. Should be one value per chain, one value per
                  parameter, or have shape (nchains,nparams)
      clevels   : list of floats between 0 and 1
                  percentiles at which to show contours
      contour_reference : one of {'samples', 'chi2'}
                  whether to draw contour on fractions of samples or
                  on likelihood levels. In the former case, *clevels*
                  must be floats between 0 and 1; in the latter, the
                  levels of the chi2. ONLY 'samples' IMPLEMENTED SO FAR
      truths    : one of {list of floats, 'medians', None}
                  reference values for each parameter, to be shown in
                  each panel
      smooth    : float
                  the width of the gaussian with which to smooth the
                  contours in the off-diagonal panels. If no value is given,
                  the contours are not smoothed.
      likelihood : array of floats
                  the likelihood surface, to be shown as a histogram in the
                  diagonals or to be used to define the 2d contours. If
                  contour_reference=='chi2' then provide the chi2 here
                  instead of the likelihood
      show_likelihood_1d : bool
                  whether to show the likelihood in the diagonal panels
      likesmooth : int
                  the number of maxima to average over to show the
                  likelihood surface
      colors    : any argument taken by the *colors* argument of
                  pylab.contour(), or a tuple of them if more than one
                  model is to be plotted
      ls1d      : one of {'solid','dashed','dashdot','dotted'}
                  linestyle for the diagonal plots, if style1d=='curve'.
                  Can specify more than one value as a list if more than one
                  model is being plotted.
      ls2d      : one of {'solid','dashed','dashdot','dotted'}
                  linestyle for the contours. Can specify more than one value
                  as a list if more than one model is being plotted.
      style1d   : one of {'bar', 'step', 'stepfilled', 'curve'}
                  if 'curve', plot the 1d posterior as a curve; else this
                  parameter is passed to the 'histtype' argument in
                  pyplot.hist()
      medians1d : bool
                  whether to show the medians in the diagonal panels as
                  vertical lines
      percentiles1d : bool
                  whether to show selected percentiles (see *clevels*) in the
                  diagonal panels as vertical lines
      background : one of {None, 'points', 'density', 'filled'}
                  If not None, then either points, a smoothed 2d histogram,
                  or filled contours are plotted beneath contours.
      bweight   : array-like, same length as e.g., A1
                  values to color-code background points
      bcolor    : color property, consistent with *background*
                  color of the points or filled contours, or colormap of the
                  2d density background.
      alpha     : float between 0 and 1
                  transparency of the points if shown
      limits    : list of length-2 lists
                  a list of plot limits for each of the parameters.
      ticks     : list of lists
                  a list of tick positions for each parameter, to be printed
                  both in the x and y axes as appropriate.
      top_labels : bool
                  whether to show axis and tick labels at the top of each
                  diagonal plot
      pad       : float
                  blank space outside axes (passed to tight_layout)
      h_pad     : float
                  vertical space between axes (passed to tight_layout)
      w_pad     : float
                  horizontal space between axes (passed to tight_layout)
      output    : string
                  filename to save the plot.
      verbose   : boolean
                  whether to print the marginalized values per variable
      names_kwargs : dictionary
                  keyword arguments controlling the location and style
                  of the legend containing model names; passed to
                  pylab.legend(). The default settings are:
                      * 'loc': 'upper right'
                      * 'frameon': False
                      * 'bbox_to_anchor': (0.95,0.95)
                      * 'bbox_transform': pylab.gcf().transFigure
      kwargs    : keyword arguments to be passed to pylab.contour()


    Returns
    -------
      fig, axes_diagonal, axes_off : pylab figure and axes (diagonal and
                  off-diagonal) instances

    """
    from numpy import append, array, digitize, exp, histogram, histogram2d
    from numpy import linspace, median, percentile, sort, transpose
    from scipy.ndimage.filters import gaussian_filter
    if style1d == 'curve':
        from scipy import interpolate

    # not yet implemented
    options = _load_corner_config(config)
    # the depth of an array or list. Useful to assess the proper format of
    # arguments. Returns zero if scalar.
    depth = lambda L: len(numpy.array(L).shape)
    #nchains = (len(X)-1 if depth(X) > 1 else 1)
    nchains = max(depth(X)-1, 1)
    if nchains > 1:
        ndim = len(X[0])
        nsamples = len(X[0][0])
        if background == 'points':
            background = None
    else:
        ndim = len(X)
        nsamples = len(X[0])
        X = (X,)
        if likelihood is not None:
            likelihood = (likelihood,)
    if nsamples == 0:
        msg = 'plottools.corner: received empty array.'
        msg += ' It is possible that you set the burn-in to be longer'
        msg += ' than the chain itself!'
        raise ValueError(msg)
    # check ticks
    if ticks is not None:
        if len(ticks) != ndim:
            print('WARNING: number of tick lists does not match' \
                  ' number of parameters')
            ticks = None
    # check limits
    if limits is not None:
        if len(limits) != ndim:
            print('WARNING: number of limit lists does not match' \
                  ' number of parameters')
            limits = None
    # check likelihood
    if likelihood is not None and show_likelihood_1d:
        msg = 'WARNING: likelihood format not right - ignoring'
        lshape = likelihood.shape
        if len(lshape) == 1:
            likelihood = [likelihood]
        if lshape[0] != nchains or lshape[1] != nsamples \
                or len(lshape) != 2:
            print(msg)
            likelihood = None

    # check clevels - they should be fractions between 0 and 1 for
    # contour_reference == 'samples'.
    #if contour_reference != 'samples':
        #msg = 'ERROR: only "samples" option implemented for'
        #msg += ' contour_reference. Setting contour_reference="samples"'
        #print(msg)
        #contour_reference = 'samples'
    if contour_reference == 'samples':
        if 1 < max(clevels) <= 100:
            clevels = [cl/100. for cl in clevels]
        elif max(clevels) > 100:
            msg = 'ERROR: contour levels must be between 0 and 1 or between'
            msg += ' 0 and 100'
            print(msg)
            exit()
    # check truths
    if truths is not None:
        if len(truths) != ndim:
            msg = 'WARNING: number of truth values does not match number'
            msg += ' of parameters'
            print(msg)
            truths = None
    try:
        if len(smooth) != len(X[0]):
            print('WARNING: number of smoothing widths must be equal to' \
                  ' number of parameters')
            smooth = [0 for i in X[0]]
    except TypeError:
        if smooth not in (False, None):
            smooth = [smooth for i in X[0]]
    # check the binning scheme.
    meta_bins = [bins, bins1d]
    for i, bname in enumerate(('bins','bins1d')):
        bi = array(meta_bins[i])
        bidepth = depth(bi)
        # will be the same message in all cases below
        msg = 'ERROR: number of {0} must equal either number'.format(bname)
        msg += ' of chains or number of parameters, or have shape'
        msg += ' (nchains,nparams)'
        # this means binning will be the same for all chains
        ones = numpy.ones((nchains,ndim))
        # is it a scalar?
        if bidepth == 0:
            meta_bins[i] = bi.T * ones
        # or a 1d list?
        elif bidepth == 1:
            bi = numpy.array(bi)
            if len(bi) == ndim:
                meta_bins[i] = ones * bi
            elif len(bi) == nchains:
                meta_bins[i] = ones * bi[:,numpy.newaxis]
            else:
                print(msg)
                exit()
        elif (bidepth == 2 and nchains > 1 and \
              numpy.array(bi).shape != ones.shape) or \
             bidepth > 2:
            print(msg)
            exit()
    # adjusted to the required shape (and type)
    bins, bins1d = meta_bins
    if isinstance(bins[0][0], float):
        bins = numpy.array(bins, dtype=int)
    if isinstance(bins1d[0][0], float):
        bins1d = numpy.array(bins1d, dtype=int)
    if len(X) == 1:
        if isinstance(colors, basestring):
            color1d = colors
        else:
            color1d = 'k'
    else:
        if len(colors) == len(X):
            color1d = colors
        # supports up to 12 names (plot would be way overcrowded!)
        else:
            color1d = ('g', 'orange', 'c', 'm', 'b', 'y',
                       'g', 'orange', 'c', 'm', 'b', 'y')
    if isinstance(ls1d, basestring):
        ls1d = [ls1d for i in X]
    if isinstance(ls2d, basestring):
        ls2d = [ls2d for i in X]
    # to move the model legend around
    names_kwargs_defaults = {'loc': 'center',
                             'frameon': False,
                             'bbox_to_anchor': (0.95,0.95),
                             'bbox_transform': pylab.gcf().transFigure}
    for key in names_kwargs_defaults:
        if key not in names_kwargs:
            names_kwargs[key] = names_kwargs_defaults[key]
    # all set!
    axvls = ('--', ':', '-.')
    fig, axes = pylab.subplots(figsize=(2*ndim+1,2*ndim+1), ncols=ndim,
                               nrows=ndim)
    # diagonals first
    plot_ranges = []
    axes_diagonal = []
    # to generate model legend
    model_lines = []
    # for backward compatibility
    histtype = style1d.replace('hist', 'step')
    for i in xrange(ndim):
        ax = axes[i][i]
        axes_diagonal.append(ax)
        peak = 0
        edges = []
        for m, Xm in enumerate(X):
            edges.append([])
            if style1d == 'curve':
                ho, e = histogram(Xm[i], bins=bins1d[m][i], normed=True)
                xo = 0.5 * (e[1:] + e[:-1])
                xn = linspace(xo.min(), xo.max(), 500)
                n = interpolate.spline(xo, ho, xn)
                line, = ax.plot(xn, n, ls=ls1d[m], color=color1d[m])
                if i == 0:
                    model_lines.append(line)
            else:
                n, e, patches = ax.hist(Xm[i], bins=bins1d[m][i],
                                        histtype=histtype,
                                        color=color1d[m], normed=True)
            edges[-1].append(e)
            if n.max() > peak:
                peak = n.max()
            area = n.sum()
            if medians1d:
                ax.axvline(median(Xm[i]), ls='-', color=color1d[m])
            if verbose:
                if len(names) == len(X):
                    print('names[{0}] = {1}'.format(m, names[m]))
                if labels is not None:
                    print('  {0}'.format(labels[i]), end=' ')
                    if truths is None:
                        print('')
                    else:
                        print('(truth: {0})'.format(truths[i]))
                    print('    p50.0  {0:.3f}'.format(median(Xm[i])))
            for p, ls in izip(clevels, axvls):
                v = [percentile(Xm[i], 100*(1-p)/2.),
                     percentile(Xm[i], 100*(1+p)/2.)]
                if percentiles1d:
                    ax.axvline(v[0], ls=ls, color=color1d[m])
                    ax.axvline(v[1], ls=ls, color=color1d[m])
                if verbose:
                    print('    p%.1f  %.3f  %.3f' %(100*p, v[0], v[1]))
        if likelihood is not None:
            for m, Xm, Lm, e in izip(count(), X, likelihood, edges):
                binning = digitize(Xm[i], e[m])
                xo = 0.5 * (e[m][1:] + e[m][:-1])
                # there can be nan's because some bins have no data
                valid = array([(len(Lm[binning == ii]) > 0)
                               for ii in xrange(1, len(e[m]))])
                Lmbinned = [median(sort(Lm[binning == ii+1])[-likesmooth:])
                            for ii, L in enumerate(valid) if L]
                # normalized to the histogram area
                Lmbinned = exp(Lmbinned)
                Lmbinned -= Lmbinned.min()
                Lmbinned /= Lmbinned.sum() / area
                ax.plot(xo[valid], Lmbinned, '-',
                        color=color_likelihood, lw=1, zorder=-10)
        if truths_in_1d and truths is not None:
            ax.axvline(truths[i], ls='-', color=truth_color,
                       zorder=10)
        if i == ndim-1 and labels is not None:
            if len(labels) >= ndim:
                ax.set_xlabel(labels[i])
        ax.set_yticks([])
        # to avoid overcrowding tick labels
        if ticks is None:
            tickloc = pylab.MaxNLocator(3)
            ax.xaxis.set_major_locator(tickloc)
        else:
            ax.set_xticks(ticks[i])
        pylab.xticks(rotation=45)
        if limits is not None:
            ax.set_xlim(*limits[i])
        ax.set_ylim(0, 1.1*peak)
        if i != ndim-1:
            ax.set_xticklabels([])
        if top_labels:
            topax = ax.twiny()
            topax.set_xlim(*ax.get_xlim())
            topax.xaxis.set_major_locator(tickloc)
            topax.set_xlabel(labels[i])
        plot_ranges.append(ax.get_xlim())

    # lower off-diagonals
    axes_off = []
    # vertical axes
    for i in xrange(1, ndim):
        # blank axes
        axes[0][i].axis('off')
        for j in xrange(i+1, ndim):
            axes[i][j].axis('off')
        # horizontal axes
        for j in xrange(i):
            ax = axes[i][j]
            axes_off.append(ax)
            extent = append(plot_ranges[j], plot_ranges[i])
            for m, Xm in enumerate(X):
                if contour_reference == 'likelihood':
                    ax.contour(Xm[j], Xm[i], likelihood, levels=clevels,
                               linewidths=1)
                    continue
                if contour_reference == 'samples':
                    h = histogram2d(Xm[j], Xm[i], bins=bins[m][i])
                    h, xe, ye = histogram2d(Xm[j], Xm[i], bins=bins[m][i])
                    h = h.T
                    extent = (xe[0], xe[-1], ye[0], ye[-1])
                    if smooth not in (False, None):
                        h = gaussian_filter(h, (smooth[i],smooth[j]))
                    levels = contour_levels(Xm[j], Xm[i], bins=bins[m][i],
                                            levels=clevels)
                if background == 'points':
                    if not (cmap is None or bweight is None):
                        ax.scatter(Xm[j], Xm[i], c=bweight, marker='.',
                                   s=4, lw=0, cmap=cmap, zorder=-10)
                    else:
                        ax.plot(Xm[j], Xm[i], ',',
                                color=bcolor, alpha=alpha, zorder=-10)
                elif background == 'density':
                    ax.imshow([Xm[i], Xm[j]], cmap=cm.Reds,
                               extent=extent)
                elif background == 'filled':
                    lvs = contour_levels(Xm[j], Xm[i], bins=bins[m][i],
                                         levels=clevels)
                    lvs = append(lvs[::-1], h.max())
                    try:
                        if hasattr(bcolor[0], '__iter__'):
                            bcolor = [bc for bc in bcolor]
                    except TypeError:
                        pass
                    for l in xrange(len(levels), 0, -1):
                        if isinstance(bcolor[l-1][0], float) and \
                                not hasattr(bcolor[l-1][0], '__iter__'):
                            bcolor[l-1] = [bcolor[l-1]]
                        ax.contourf(h, (lvs[l-1],lvs[l]),
                        #ax.contourf(h, (lvs[l],lvs[l-1]),
                                    extent=extent, colors=bcolor[l-1])
                if show_contour:
                    ax.contour(h, levels[::-1], colors=color1d[m],
                               linestyles=ls2d[m], extent=extent,
                               zorder=10, **kwargs)
                if truths is not None:
                    #pylab.axvline(truths[j], ls='-', color=(0,0.5,1))
                    #pylab.axhline(truths[i], ls='-', color=(0,0.5,1))
                    ax.plot(truths[j], truths[i], '+',
                            color=truth_color, mew=4, ms=12, zorder=10)
            if labels is not None:
                if len(labels) == ndim:
                    if j == 0:
                        ax.set_ylabel(labels[i])
                    if i == ndim - 1:
                        ax.set_xlabel(labels[j])
            if j > 0:
                ax.set_yticklabels([])
            if i < ndim - 1:
                ax.set_xticklabels([])
            ax.set_xlim(*plot_ranges[j])
            ax.set_ylim(*plot_ranges[i])
            if ticks is not None:
                ax.set_xticks(ticks[j])
                ax.set_yticks(ticks[i])
            else:
                # to avoid overcrowding tick labels
                ax.xaxis.set_major_locator(pylab.MaxNLocator(3))
                ax.yaxis.set_major_locator(pylab.MaxNLocator(3))
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
    if (len(X) == 1 and isinstance(names, basestring)) or \
            (hasattr(names, '__iter__') and len(names) == len(X)):
        fig.legend(model_lines, names, **names_kwargs)
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if output:
        pylab.savefig(output, format=output[-3:])
        pylab.close()
    return fig, axes_diagonal, axes_off


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
        array = numpy.linspace(vmin, vmax, n)
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
        fig = pylab
    if tight:
        fig.tight_layout(**tight_kwargs)
    fig.savefig(output)
    if close:
        pylab.close()
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


def _load_corner_config(config):
    """
    Not implemented!

    """
    options = {}
    # is there a configuration file at all!?
    if config is None:
        return options
    data = numpy.loadtxt(config, dtype=str, unpack=True)
    for key, value in izip(*data):
        values = value.split(';')
        ndim = len(values)
        values = [val.split(',') for val in values]
        for i in xrange(ndim):
            for j in xrange(len(values)):
                try:
                    values[i][j] = float(values[i][j])
                except ValueError:
                    pass
                try:
                    values[i][j] = int(values[i][j])
                except ValueError:
                    pass
        if ndim == 1:
            values = values[0]
        options[key] = values
    return options

