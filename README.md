# plottools
Plotting tools

This module is focused mostly on two functions: corner() and wcslabels(). Both have explanatory help files but briefly,

    plottools.corner()

 * Make a corner plot, typically to view the output of an MCMC (see figure 6 of http://adsabs.harvard.edu/abs/2015MNRAS.454.3938S for a plot made with this function).
 * Can handle as many parameters as desired, multiple chains, and has many options to tailor the style to the user and the problem at hand.

    plottools.wcslabels()

 * Create right ascension and declination ticks in hms and dms formats, respectively, given axis limits in decimal degrees.
 
 
