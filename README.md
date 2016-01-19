# plottools
(Not so) Generic plotting tools

To install, clone this repository to somewhere in your python path by doing:

    git clone git@github.com:cristobal-sifon/plottools.git

This module is focused mostly on two functions: ``plottools.corner`` and ``plottools.wcslabels``. Both have explanatory help files but briefly,

    plottools.corner()

 * Make a corner plot, typically to view the output of an MCMC. It can handle as many parameters as desired, multiple chains, and has many options to tailor the style to the user and the problem at hand. Additionally, it returns the Figure and Axes objects, so the user can add features at will on top of the corner plot itself. For an example corner plot see figure 6 of [Sifon et al. (2015)] (http://arxiv.org/abs/1507.00737).

    plottools.wcslabels()

 * Create right ascension and declination ticklabels in hms and dms formats, respectively, given axis limits in decimal degrees.
