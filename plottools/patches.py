from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import patches as mpl_patches
from matplotlib.patches import Polygon
from matplotlib import docstring


class Bracket(Polygon):

    def __str__(self):
        pars = (self.center[0], self.center[1],
                self.length, self.width, self.angle)
        fmt = "Bracket(xy=({0}, {1}), length={2}, width={3}, angle={4}"
        return fmt.format(*pars)

    @docstring.dedent_interpd
    def __init__(self, xy, length, width, angle, **kwargs):
        """
        *xy*
          center of bracket

        *length*
          length of bracket

        *width*
          width of bracket, i.e., length of ticks at the ends.
          The width can be negative, in which case the brackets will
          extend in the opposite direction. A positive width means that
          the ticks will point upwards for an angle=0.

        *angle*
          rotation in degrees (anti-clockwise)

        Note that the facecolor is set to 'none'; if passed it will be
        ignored.

        Valid kwargs are:
        %(Patch)s
        """
        self.center = xy
        self.length = length
        self.width = width
        self.angle = angle
        rad = np.pi/180 * self.angle
        x1 = [self.center[0] + np.cos(rad)*length/2,
              self.center[1] + np.sin(rad)*length/2]
        x2 = [self.center[0] - np.cos(rad)*length/2,
              self.center[1] - np.sin(rad)*length/2]
        xd1 = [x1[0]-width*np.sin(rad), x1[1]+width*np.cos(rad)]
        xd2 = [x2[0]-width*np.sin(rad), x2[1]+width*np.cos(rad)]
        if 'facecolor' in kwargs:
            kwargs.pop('facecolor')
        if 'fc' in kwargs:
            kwargs.pop('fc')
        super().__init__(
            [xd1, x1, x2, xd2], closed=False, fc='none', **kwargs)
