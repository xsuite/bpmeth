import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CanvasZX:
    def __init__(self, fig=None, ax=None):
        """
        Canvas on which trajectories can be drawn, in ZX plane.
        :param fig: Optional matplotlib figure to use. If None, a new figure will be created.
        :param ax: Optional matplotlib axes to use. If None, new axes will be created.
        """
        
        if fig is None:
            fig = plt.figure()
        if ax is None:
            self.fig = fig
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("$z$")
            self.ax.set_ylabel("$x$")
            self.ax.set_aspect("equal")
            self.ax.grid(True)
        else:
            self.ax = ax
            self.fig = ax.figure

    def plot(self, vv, **kwargs):
        """ 
        :param vv: Array of shape (3, N) containing the points to plot, containing x, y, z.
        """
        
        self.ax.plot(vv[2], vv[0], **kwargs)

    def arrow(self, vv, dd, **kwargs):
        """
        Draw an arrow
        :param vv: Starting point of the arrow, as a 3D vector (x, y, z).
        :param dd: Direction and length of the arrow, as a 3D vector (dx, dy, dz). 
        """
        
        arr = np.hypot(dd[2], dd[0]) * 0.05
        self.ax.arrow(vv[2], vv[0], dd[2], dd[0], width=arr, **kwargs)
        

class CanvasZXY:
    def __init__(self, fig=None, ax=None):
        """
        Canvas on which trajectories can be drawn, in ZXY plane.
        :param fig: Optional matplotlib figure to use. If None, a new figure will be created.
        :param ax: Optional matplotlib axes to use. If None, new axes will be created.
        """

        if fig is None:
            fig = plt.figure()
        if ax is None:
            self.fig = fig
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel("$z$")
            self.ax.set_ylabel("$x$")
            self.ax.set_zlabel("$y$")
        else:
            self.ax = ax
            self.fig = ax.figure

    def plot(self, vv, **kwargs):
        """
        :param vv: Array of shape (3, N) containing the points to plot, containing x, y, z.
        """
        
        self.ax.plot(vv[2], vv[0], vv[1], **kwargs)

    def arrow(self, vv, dd, **kwargs):
        """ 
        Draw an arrow
        :param vv: Starting point of the arrow, as a 3D vector (x, y, z).
        :param dd: Direction and length of the arrow, as a 3D vector (dx, dy, dz).
        """
        
        self.ax.quiver(vv[2], vv[0], vv[1], dd[2], dd[0], dd[1], arrow_length_ratio=0.2, **kwargs)


class Frame:
    def __init__(self, matrix=None):
        """ 
        :param matrix: Optional 4x4 matrix specifying the frame. 
        matrix[0:3, 0], matrix[0:3, 1], matrix[0:3, 2] are the x, y, z direction vectors of the frame, and
        matrix[0:3, 3] is the origin of the frame. matrix[3] is unused.
        If None, the identity matrix will be used, corresponding to a frame at the origin with standard basis vectors.
        """ 
        
        if matrix is None:
            self.matrix = np.eye(4, 4)
        else:
            self.matrix = matrix

    @property
    def origin(self):
        return self.matrix[0:3, 3]

    @origin.setter
    def origin(self, value):
        self.matrix[0:3, 3] = value

    @property
    def rotation(self):
        return self.matrix[0:3, 0:3]

    @rotation.setter
    def rotation(self, value):
        self.matrix[0:3, 0:3] = value

    @property
    def xdir(self):
        return self.matrix[0:3, 0]

    @property
    def ydir(self):
        return self.matrix[0:3, 1]

    @property
    def zdir(self):
        return self.matrix[0:3, 2]


    def plot_zx(self, canvas=None, arrowsize=0.05):
        """
        Plot the frame on a ZX canvas, with arrows indicating the x and z directions.
        :param canvas: Optional CanvasZX object to plot on. If None, a new CanvasZX will be created.
        :param arrowsize: Size of the arrows indicating the x and z directions.
        """
        
        if canvas is None:
            canvas = CanvasZX()
        x = self.matrix[0, 3]
        z = self.matrix[2, 3]
        canvas.plot(self.origin, linestyle="none", marker="o", color="black")
        canvas.arrow(self.origin, self.xdir*arrowsize, color="red")
        canvas.arrow(self.origin, self.zdir*arrowsize, color="blue")
    
    
    def plot_zxy(self, canvas=None, arrowsize=0.05):
        """
        Plot the frame on a ZXY canvas, with arrows indicating the x, y, and z directions.
        :param canvas: Optional CanvasZXY object to plot on. If None, a new CanvasZXY will be created.
        :param arrowsize: Size of the arrows indicating the x, y, and z directions.
        """
        
        if canvas is None:
            canvas = CanvasZXY()
        x = self.matrix[0, 3]
        y = self.matrix[1, 3]
        z = self.matrix[2, 3]
        canvas.plot(self.origin, linestyle="none", marker="o", color="black")
        canvas.arrow(self.origin, self.xdir*arrowsize, color="red")
        canvas.arrow(self.origin, self.ydir*arrowsize, color="orange")
        canvas.arrow(self.origin, self.zdir*arrowsize, color="blue")


    def move_to(self, origin):
        """
        Move the frame to a new origin, without changing its orientation.
        :param origin: New origin of the frame, as a 3D vector (x, y, z). 
        """
        
        self.origin = origin
        
        return self


    def move_by(self, offset):
        """ 
        Move the frame by a given offset, without changing its orientation.
        :param offset: Offset to move the frame by, as a 3D vector (dx, dy, dz).
        """
        
        self.origin += offset
        
        return self


    def rotate_x(self, angle):
        """ 
        Rotate the frame around the x-axis by a given angle around its origin.
        :param angle: Angle to rotate the frame by, in radians. 
        """
        
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        self.rotation = np.matmul(rot, self.rotation)

        return self


    def rotate_y(self, angle):
        """ 
        Rotate the frame around the y-axis by a given angle around its origin.
        :param angle: Angle to rotate the frame by, in radians. 
        """
        
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.rotation = np.matmul(rot, self.rotation)
        
        return self


    def rotate_z(self, angle):
        """ 
        Rotate the frame around the z-axis by a given angle around its origin.
        :param angle: Angle to rotate the frame by, in radians. 
        """
        
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.rotation = np.matmul(rot, self.rotation)
        
        return self


    def arc_by(self, length, angle):
        """
        Move the frame along an arc of a circle with given length and angle, in the plane defined by the x and z directions of the frame.
        :param length: Length of the arc to move along.
        :param angle: Angle of the arc to move along, in radians. 
        """
        
        if angle == 0:
            return self.move_by([0, 0, length])
        rho = length / angle
        self.origin -= rho * self.xdir
        self.rotate_y(-angle)
        self.origin += rho * self.xdir
        
        return self


    def copy(self):
        return Frame(self.matrix.copy())


    def trajectory(self, s,x,y):
        """ 
        Return the global coordinates of a trajectory with coordinates x,y,s in the frame.
        """
        
        qq=np.zeros((3,len(s)))
        for ii, ss in enumerate(s):
            qq[:,ii]=(self.matrix@np.array([x[ii],y[ii],s[ii],1]))[:3]
        return qq


    def ref_trajectory(self, steps=21, smin=0, smax=1):
        """ 
        Return the reference trajectory in global coordinates, given by x=y=0 in local coordinates.
        """
        
        s = np.linspace(smin, smax, steps)
        x = np.zeros(steps)
        y = np.zeros(steps)
        return self.trajectory(s,x,y)


    def plot_trajectory_zx(self, s,x,y, canvas=None, figname=None, **kwargs):
        """ 
        Plot the trajectory with coordinates x,y,s in the local frame.
        :param s: 1D array of s coordinates of the trajectory in the local frame.
        :param x: 1D array of x coordinates of the trajectory in the local frame.
        :param y: 1D array of y coordinates of the trajectory in the local frame.
        :param canvas: Optional CanvasZX object to plot on. If None, a new CanvasZX will be created.
        :param figname: Optional filename to save the figure to. If None, the figure will not be saved.
        :param **kwargs: Additional keyword arguments to pass to the plot function of the canvas, such as color, linestyle, etc.
        """
        
        if canvas is None:
            canvas = CanvasZX()
        self.plot_zx(canvas)
        canvas.plot(self.trajectory(s,x,y), **kwargs)

        if figname:
            canvas.fig.savefig(figname, dpi=500)

        return self


class BendFrame:
    def __init__(self, start, length=0, angle=0):
        """ 
        :param start: Frame at the start of the bend, as a Frame object.
        :param length: Length of the bend.
        :param angle: Angle of the bend, in radians.
        """
        
        self.start = start
        self.angle = angle
        self.length = length

    @property
    def end(self):
        """ 
        Return the end Frame object.
        """ 
        
        end = self.start.copy()
        end.arc_by(self.length, self.angle)
        return end
    
    def frame(self, s):
        """
        Return the Frame object at position s along the bend, where s=0 corresponds to the start and s=length corresponds to the end.
        """
        
        return self.start.copy().arc_by(s, s / self.length * self.angle)


    def ref_trajectory(self, steps=21):
        """
        Return the reference trajectory in global coordinates, given by x=y=0 in local coordinates.
        :param steps: Number of points to return along the trajectory, including the start and end points.
        """
        
        return np.array(
            [self.frame(s).origin for s in np.linspace(0, self.length, steps)]
        ).T


    def trajectory(self, s,x,y):
        """
        Return the given trajectory with coordinates x,y,s in the local frame, in global coordinates.
        """
        
        qq=np.zeros((3,len(s)))
        for ii,ss in enumerate(s):
            fr=self.frame(ss)
            qq[:,ii]=(fr.matrix@np.array([x[ii],y[ii],0,1]))[:3]
        return qq


    def plot_zx(self, canvas=None):
        """ 
        Plot the start and end frames of the bend, as well as the reference trajectory, on a ZX canvas.
        :param canvas: Optional CanvasZX object to plot on. If None, a new CanvasZX will be created.
        """
        
        if canvas is None:
            canvas = CanvasZX()
        self.start.plot_zx(canvas, arrowsize=self.length/10)
        self.end.plot_zx(canvas, arrowsize=self.length/10)
        canvas.plot(self.ref_trajectory(), color="green")


    def plot_zxy(self, canvas=None):
        """ 
        Plot the start and end frames of the bend, as well as the reference trajectory, on a ZXY canvas.
        :param canvas: Optional CanvasZXY object to plot on. If None, a new CanvasZXY will be created.
        """ 
        
        if canvas is None:
            canvas = CanvasZXY()
        self.start.plot_zxy(canvas, arrowsize=self.length/10)
        self.end.plot_zxy(canvas, arrowsize=self.length/10)
        canvas.plot(self.ref_trajectory(), color="green")
        

    def plot_trajectory_zx(self, s,x,y, canvas=None, figname=None, **kwargs):
        """ 
        Plot the given trajectory on a ZX canvas, along with the start and end frames and the reference trajectory.
        :param s: 1D array of s coordinates of the trajectory in the local frame.
        :param x: 1D array of x coordinates of the trajectory in the local frame.
        :param y: 1D array of y coordinates of the trajectory in the local frame.
        :param canvas: Optional CanvasZX object to plot on. If None, a new CanvasZX will be created.
        :param figname: Optional filename to save the figure to. If None, the figure will not be saved.
        """
        
        if canvas is None:
            canvas = CanvasZX()
        self.plot_zx(canvas)
        canvas.plot(self.trajectory(s,x,y), **kwargs)

        if figname:
            canvas.fig.savefig(figname, dpi=500)
            

    def plot_trajectory_zxy(self, s,x,y, canvas=None, figname=None, **kwargs):
        """ 
        Plot the given trajectory on a ZXY canvas, along with the start and end frames and the reference trajectory.
        :param s: 1D array of s coordinates of the trajectory in the local frame.
        :param x: 1D array of x coordinates of the trajectory in the local frame.
        :param y: 1D array of y coordinates of the trajectory in the local frame.
        :param canvas: Optional CanvasZXY object to plot on. If None, a new CanvasZXY will be created.
        :param figname: Optional filename to save the figure to. If None, the figure will not be saved.
        """
        
        if canvas is None:
            canvas = CanvasZXY()
        self.plot_zxy(canvas)
        canvas.plot(self.trajectory(s,x,y), **kwargs)

        # Set equal intervals for the three axes.
        smid, xmid, ymid = (np.max(s)+np.min(s))/2, (np.max(x)+np.min(x))/2, (np.max(y)+np.min(y))/2
        sdist, xdist, ydist = (np.max(s)-np.min(s))/2, (np.max(x)-np.min(x))/2, (np.max(y)-np.min(y))/2
        dist = max([sdist, xdist, ydist])
        canvas.ax.set_xlim(smid-dist, smid+dist)
        canvas.ax.set_ylim(xmid-dist, xmid+dist)
        canvas.ax.set_zlim(ymid-dist, ymid+dist)
        canvas.ax.set_box_aspect((1,1,1))
        
        if figname:
            canvas.fig.savefig(figname, dpi=500)



    
