from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.patches import Ellipse, Polygon
from matplotlib import docstring


class Bracket(Polygon):
    """
    A square bracket open polygon
    """
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
        # make sure facecolor is set to 'none'
        if 'facecolor' in kwargs:
            kwargs.pop('facecolor')
        if 'fc' in kwargs:
            kwargs.pop('fc')
        super().__init__(
            [xd1, x1, x2, xd2], closed=False, fc='none', **kwargs)


class LogEllipse(Polygon):
    """
    A scale-free ellipse meant to be drawn on a log-log plot.

    Note that the width and height do not seem to scale as expected;
    one needs to play around with them to get the appropriate shape.
    """
    def __str__(self):
        pars = (self.center[0], self.center[1],
                self.logwidth, self.logheight, self.angle)
        fmt = "LogEllipse(xy=(%s, %s), logwidth=%s, logheight=%s, angle=%s)"
        return fmt % pars

    @docstring.dedent_interpd
    def __init__(self, xy, logwidth, logheight, angle=0.0, npts=100, **kwargs):
        """
        *xy*
          center of ellipse in linear space

        *width*
          total length (diameter) of horizontal axis, in dex

        *height*
          total length (diameter) of vertical axis, in dex

        *angle*
          rotation in degrees (anti-clockwise)

        *npts*
          number of points used to sample the polygon

        Valid kwargs are:
        %(Patch)s
        """
        self.center = xy
        self.angle = angle
        self.npts = npts
        self.logx = np.log10(xy[0])
        self.logy = np.log10(xy[1])
        self.logwidth = logwidth
        self.logheight = logheight
        _x = np.linspace(-self.logwidth, self.logwidth, self.npts)
        _y = (self.logheight/self.logwidth) * (self.logwidth**2-_x**2)**0.5
        self._x = np.append(_x, _x[::-1])
        self._y = np.append(-_y, _y)
        self._rotate()
        self.pts = np.transpose(
            [self._x+self.logx, self._y+self.logy])[np.isfinite(self._y)]
        self.pts = 10**self.pts
        super().__init__(self.pts, **kwargs)

    def _rotate(self):
        rad = np.pi / 180 * self.angle
        rotmatrix = np.array([[np.cos(rad), -np.sin(rad)],
                              [np.sin(rad), np.cos(rad)]])
        pts = np.array([self._x, self._y])
        self._x, self._y = np.dot(rotmatrix, pts)
