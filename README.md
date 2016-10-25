# plottools
(Not so) Generic plotting tools

To install, download the latest release to somewhere in your python path from here:

    https://github.com/cristobal-sifon/plottools/releases

To get started, open an `(I)Python` terminal and type

    import plottools

This module is focused mostly on two functions: ``plottools.corner`` and ``plottools.wcslabels``. Both have explanatory help files. To see them, type

    help(plottools.corner)

or 

    help(plottools.wcslabels)

Briefly,

`plottools.corner` makes a corner plot, typically to view the output of an MCMC. It can handle as many parameters as desired, multiple chains, and has many options to tailor the style to the user and the problem at hand. Additionally, it returns the Figure and Axes objects, so the user can add features at will on top of the corner plot itself. For an example corner plot see figure 6 of [Sifon et al. (2015)] (http://arxiv.org/abs/1507.00737).

`plottools.wcslabels` creates right ascension and declination ticklabels in hms and dms formats, respectively, given axis limits in decimal degrees.

There are other, less used and less broad functions as well -- see the release page or type:

    help(plottools)

---

### To-do list
This is a wish list for updates to the code. Feel free to make suggestions.

  * `corner()`:
    * use different `bcolors` for different models

---
*Last updated: 2016-Oct-25*

*(c) Cristóbal Sifón (Princeton)*
