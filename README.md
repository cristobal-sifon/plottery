# plottery
(Not so) Generic plotting tools

To install, run

    pip install plottery

or simply clone the latest version from github:

    git clone https://github.com/cristobal-sifon/plottery.git

The `plottery` package contains four modules, `astroplots`, `patches`, 
`plotutils`, and `statsplots`

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


## Changelog

* v0.6.6 (Dec 2022):
    - `statsplots.corner` supports strings in `bins` and `bins1d` for automatic bin width calculation
* v0.6.5 (Dec 2022):
    - `statsplots.corner` bug fix when attempting to plot likelihood in diagonal panels (#27)


---
*Last updated: Dec 2022*

*(c) Cristóbal Sifón 2013-2022*
