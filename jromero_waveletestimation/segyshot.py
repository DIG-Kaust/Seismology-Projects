import numpy as np
import matplotlib.pyplot as plt

import segyio


def rotate(x, y, ox, oy, angle):
    """Rotation

    Apply rotation matrix as in https://en.wikipedia.org/wiki/Rotation_matrix

    Parameters
    ----------
    x : :obj:`np.ndarray`
        x-coordinates
    y : :obj:`np.ndarray`
        y-coordinates
    ox : :obj:`float`
        x origin
    oy : :obj:`float`
        y origin
    angle : :obj:`float`
        Rotation angle in radiants (taken from x-axis)

    Returns
    -------
    x : :obj:`np.ndarray`
        Rotated x-coordinates
    y : :obj:`np.ndarray`
        Rotated y-coordinates

    """
    xrot, yrot = ((x - ox) * np.cos(angle) - (y - oy) * np.sin(angle)), \
                 ((x - ox) * np.sin(angle) + (y - oy) * np.cos(angle))
    return xrot, yrot


class SegyShot:
    """Segy Shot data

    Class for manipulating single Segy file with Shot gather configuration

    Parameters
    ----------
    filename : :obj:`str`
        Name of file to read
    components : :obj:`list`, optional
        List of components recorded in data (data is assumed to be
        have consecutive traces representing different components)

    """

    def __init__(self, filename, components=['P']):

        self.filename = filename
        self.components = components
        self.ncomponents = len(components)

        with segyio.open(filename, "r", ignore_geometry=True) as sgy:
            self.ntraces_per_shot = sgy.bin[segyio.BinField.Traces]
            self.nrec = self.ntraces_per_shot // self.ncomponents
            self.nsrc = sgy.tracecount // self.ntraces_per_shot
            self.nt = sgy.bin[segyio.BinField.Samples]
        self.selected_rec = np.arange(self.nrec)
        self.selected_src = np.arange(self.nsrc)

    def interpret(self):
        """Interpret

        Scan and interpret a Seg-Y file assumed to contain shot gathers
        """
        with segyio.open(self.filename, "r", ignore_geometry=True) as sgy:
            # Time axis
            self.t = sgy.samples / 1000
            self.dt = self.t[1] - self.t[0]

            # Scaling factor
            self.sc = sgy.header[0][segyio.TraceField.SourceGroupScalar]
            if (self.sc < 0):
                self.sc = 1. / abs(self.sc)

            # Receiver locations
            self.recx = self.sc * sgy.attributes(segyio.TraceField.GroupX)[
                                  :self.nrec * self.ncomponents:self.ncomponents]
            self.recy = self.sc * sgy.attributes(segyio.TraceField.GroupY)[
                                  :self.nrec * self.ncomponents:self.ncomponents]
            self.recz = self.sc * sgy.attributes(
                segyio.TraceField.GroupWaterDepth)[
                                  :self.nrec * self.ncomponents:self.ncomponents]

            # Source locations
            self.srcx = self.sc * sgy.attributes(segyio.TraceField.SourceX)[
                                  ::self.nrec * self.ncomponents]
            self.srcy = self.sc * sgy.attributes(segyio.TraceField.SourceY)[
                                  ::self.nrec * self.ncomponents]
            self.srcz = self.sc * sgy.attributes(
                segyio.TraceField.SourceDepth)[::self.nrec * self.ncomponents]

    def showgeometry(self, local=False, onlyselected=False,
                     figsize=(10, 10), newfig=True):
        """Visualize geometry

        Visualize the acquisition geometry

        Parameters
        ----------
        local : :obj:`bool`, optional
            Local or global geometry
        onlyselected : :obj:`bool`, optional
            Show only selected sources and receivers
        figsize : :obj:`list`, optional
            Figure size
        newfig : :obj:`bool`, optional
            Create new figure or display on current one

        """
        if local:
            recx, recy = self.recx_local, self.recy_local
            srcx, srcy = self.srcx_local, self.srcy_local
        else:
            recx, recy = self.recx, self.recy
            srcx, srcy = self.srcx, self.srcy

        if newfig == True:
            plt.figure(figsize=figsize)
            if not onlyselected:
                plt.scatter(srcx, srcy, c=np.arange(self.nsrc),
                            cmap='jet', s=1, label='src')
            plt.scatter(srcx[self.selected_src], srcy[self.selected_src],
                        color='r', s=20, label='selected src')
            if not onlyselected:
                plt.scatter(recx, recy, color='b', s=1, label='rec')
            plt.scatter(recx[self.selected_rec], recy[self.selected_rec],
                        color='b', s=20, label='selected rec')
            plt.legend()

        else:
            if not onlyselected:
                plt.scatter(srcx, srcy, s=1, color='r', label='src')
            plt.scatter(srcx[self.selected_src], srcy[self.selected_src],
                        color='r', s=20, label='selected src')
            if not onlyselected:
                plt.scatter(recx, recy, s=1, color='b', label='rec')
            plt.scatter(recx[self.selected_rec], recy[self.selected_rec],
                        s=20, color='b', label='selected rec')
            plt.legend()
            

            
            
        with np.printoptions(threshold=np.inf):
            print('Sources X\n',srcx[self.selected_src],'\n')
            print('Sources Y\n',srcy[self.selected_src],'\n')
#             print(srcy[self.selected_src].shape)
            print('Receivers X\n',recx[self.selected_rec],'\n')
            print('Receivers Y\n',recy[self.selected_rec],'\n')
#             print(srcx[self.selected_rec].shape)
#             print(srcx[self.selected_src]-recx[self.selected_rec][::2])

        
        
    def rotategeometry(self, velfile, rotation=None, plotflag=0):
        """Rotate geometry

        Rotate the acquisition geometry based on a velocity model

        Parameters
        ----------
        velfile : :obj:`str`
            Filename of velocity model to use for rotation
        r : :obj:`list`, optional
            Rotation as [rot, ox, oy] (if provided, will not be
            inferred from velfile)
        plotflag : :obj:`bool`, optional
            Plot intermediate results

        """
        # read velocity file
        with segyio.open(velfile, "r") as vel:
            scvel = vel.header[0][segyio.TraceField.SourceGroupScalar]
            if scvel < 0:
                scvel = 1. / abs(scvel)
            xvel = scvel * vel.attributes(segyio.TraceField.CDP_X)[:]
            yvel = scvel * vel.attributes(segyio.TraceField.CDP_Y)[:]

            oxvel, oyvel = scvel * vel.attributes(segyio.TraceField.CDP_X)[0], \
                           scvel * vel.attributes(segyio.TraceField.CDP_Y)[0]
            oxvel1, oyvel1 = scvel * vel.attributes(segyio.TraceField.CDP_X)[
                len(vel.xlines) - 1], \
                             scvel * vel.attributes(segyio.TraceField.CDP_Y)[
                                 len(vel.xlines) - 1]

            # find and apply rotation
            if rotation is not None:
                self.rot, self.ox, self.oy = rotation
            else:
                rot, ox, oy = segyio.tools.rotation(vel, line='fast')
                self.ox, self.oy = scvel * ox, scvel * oy
                self.rot = (rot - np.pi / 2)

            xvel_local, yvel_local = rotate(xvel, yvel, self.ox, self.oy,
                                            self.rot)
            oxvel_local, oyvel_local = rotate(oxvel, oyvel, self.ox, self.oy,
                                              self.rot)
            oxvel1_local, oyvel1_local = rotate(oxvel1, oyvel1, self.ox,
                                                self.oy, self.rot)

            # rotated sources and receivers
            self.srcx_local, self.srcy_local = rotate(self.srcx, self.srcy,
                                                      self.ox, self.oy,
                                                      self.rot)
            self.recx_local, self.recy_local = rotate(self.recx, self.recy,
                                                      self.ox, self.oy,
                                                      self.rot)

            # identify local regular axis
            xextent = np.max(xvel_local) - np.min(xvel_local)
            yextent = np.max(yvel_local) - np.min(yvel_local)
            dx = xextent / len(vel.xlines)
            dy = yextent / len(vel.ilines)

            print(
                'Local regular axis:\n ox=%f, dx=%f nx=%d\n oy=%f, dy=%f ny=%d'
                % (oxvel_local, dx, len(vel.xlines), oyvel_local, dy,
                   len(vel.ilines)))

            if plotflag:
                # Display rotated geometry
                plt.figure()
                plt.scatter(xvel, yvel, color='k', label='Velocity model')
                plt.scatter(self.recx, self.recy, color='b', label='rec')
                plt.scatter(self.srcx, self.srcy, color='r', label='src')
                plt.scatter(oxvel, oyvel, color='c', label='IL=0, XL=0')
                plt.scatter(oxvel1, oyvel1, color='y', label='IL=0, XL=end')
                plt.legend()

                plt.figure()
                plt.scatter(xvel_local, yvel_local, color='k',
                            label='Velocity model')
                plt.scatter(self.recx_local, self.recy_local, color='b',
                            label='rec')
                plt.scatter(self.srcx_local, self.srcy_local, color='r',
                            label='src')
                plt.scatter(oxvel_local, oyvel_local, color='c',
                            label='IL=0, XL=0')
                plt.scatter(oxvel1_local, oyvel1_local, color='y',
                            label='IL=0, XL=end')
                plt.legend()

            return dx, len(vel.xlines), dy, len(vel.ilines)

    def resetrecs(self):
        """Reset selection of subset of receivers
        """
        self.selected_rec = np.arange(self.nrec)

    def resetsrcs(self):
        """Reset selection of subset of sources
        """
        self.selected_src = np.arange(self.nsrc)

    def selectrecs(self, start=0, end=-1, plotflag=0):
        """Select subset of receivers

        Parameters
        ----------
        start : :obj:`int`, optional
            Index of first receiver to select
        end : :obj:`int`, optional
            Index of last receiver to select
        plotflag : :obj:`int`, optional
            Plot intermediate results

        """
        self.resetrecs()
        if end == -1:
            end = self.nrec
        self.selected_rec = self.selected_rec[start:end:8]# FIX TO 8 inc

        if plotflag:
            self.showgeometry()
            plt.legend()

    def selectsrcs(self, start=0, end=-1, plotflag=0):
        """Select subset of sources

        Parameters
        ----------
        start : :obj:`int`, optional
            Index of first source to select
        end : :obj:`int`, optional
            Index of last source to select
        plotflag : :obj:`int`, optional
            Plot intermediate results

        """
        self.resetsrcs()
        if end == -1:
            end = self.nsrc
        self.selected_src = self.selected_src[start:end:8]

        if plotflag:
            self.showgeometry()
            plt.legend()

    def get_shotgather(self, isrc):
        """Retrieve shot gather

        Parameters
        ----------
        isrc : :obj:`int`
            Source index

        Returns
        -------
        shot : :obj:`dict`
            Shot gather (each component is an element of the returned dictionary
        """
        with segyio.open(self.filename, "r", ignore_geometry=True) as sgy:
            d = segyio.collect(sgy.trace[isrc * self.ntraces_per_shot:(
                                                                              isrc + 1) * self.ntraces_per_shot])
        shot = {component: d[ic::self.ncomponents] for ic, component in
                enumerate(self.components)}
        return shot
