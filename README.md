# plottools
(Not so) Generic plotting tools

To install, download the latest release to somewhere in your python path from here:

    https://github.com/cristobal-sifon/plottools/releases

or simply clone the latest version:

    git clone https://github.com/cristobal-sifon/plottools.git

The `plottools` package contains three modules, `astroplots`, `plotutils`, and `statsplots`, in addition to `colormaps`, which includes the new `matplotlib` colors, and was written by Nathaniel J. Smith, Stefan van der Walt, and (in the case of viridis) Eric Firing. See https://github.com/BIDS/colormap.

Below is a brief description of each module's functions. See their help pages for more details.

    astroplots:
        contour_overlay -- Overlay contours from one image on to another (new in v0.3.1).
        phase_space -- Plot phase space diagram (i.e., velocity vs. distance).
        wcslabels -- Generate HMS and DMS labels for RA and Dec given in decimal degrees.
    plotutils:
        colorscale -- Generate a colorbar and associated array of colors from a given data set.
        savefig -- Convenience wrapper around functions used when commonly saving a figure.
        update_rcParams -- Update rcParam configuration to make plots look nicer.
    statsplots:
        contour_levels -- Calculate contour levels at chosen percentiles for 2-dimensional data.
        contours -- Plot contours using a reference image with WCS coordinates.
        corner -- Make a corner plot.
        

---

### To-do list
This is a wish list for updates to the code. Feel free to make suggestions.

  * `statsplots.corner()`:
    * use different `bcolors` for different models
    * implement different `truths` for different models

---
*Last updated: Feb 2018*

*(c) Cristóbal Sifón (Princeton)*
