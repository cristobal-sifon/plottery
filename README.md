# plott
(Not so) Generic plotting tools

To install, run

    pip install plott

or simply clone the latest version from github:

    git clone https://github.com/cristobal-sifon/plott.git

The `plott` package contains four modules, `astroplots`, `patches`, `plotutils`, 
and `statsplots`, in addition to `colormaps`, which includes the new 
`matplotlib` colors, and was written by Nathaniel J. Smith, Stefan van der Walt, 
and (in the case of viridis) Eric Firing. See https://github.com/BIDS/colormap.

Below is a brief description of each module's functions. See their help pages for more details.

    astroplots:
        contour_overlay -- Overlay contours from one image on to another (new in v0.3.1).
        phase_space -- Plot phase space diagram (i.e., velocity vs. distance).
        wcslabels -- Generate HMS and DMS labels for RA and Dec given in decimal degrees.
    patches: additional matplotlib.patches objects
        Bracket -- a square bracket used to highlight a region in a figure.
        LogEllipse -- a finely-sampled polygon that appears as an ellipse in a log-log plot.
    plotutils:
        colorscale -- Generate a colorbar and associated array of colors from a given data set.
        savefig -- Convenience wrapper around functions used when commonly saving a figure.
        update_rcParams -- Update rcParam configuration to make plots look nicer.
    statsplots:
        contour_levels -- Calculate contour levels at chosen percentiles for 2-dimensional data.
        corner -- Make a corner plot.

---
*Last updated: Jan 2021*

*(c) Cristóbal Sifón 2013-2021*
