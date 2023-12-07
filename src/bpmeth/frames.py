import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CanvasZX:
    def __init__(self, fig=None, ax=None):
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
        self.ax.plot(vv[2], vv[0], **kwargs)

    def arrow(self, vv, dd, **kwargs):
        arr = np.hypot(dd[2], dd[0]) * 0.05
        self.ax.arrow(vv[2], vv[0], dd[2], dd[0], width=arr, **kwargs)
        

class CanvasZXY:
    def __init__(self, fig=None, ax=None):
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
        self.ax.plot(vv[2], vv[0], vv[1], **kwargs)

    def arrow(self, vv, dd, **kwargs):
        self.ax.quiver(vv[2], vv[0], vv[1], dd[2], dd[0], dd[1], arrow_length_ratio=0.2, **kwargs)


class Frame:
    def __init__(self, matrix=None):
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

    def plot_zx(self, canvas=None):
        if canvas is None:
            canvas = CanvasZX()
        x = self.matrix[0, 3]
        z = self.matrix[2, 3]
        canvas.plot(self.origin, linestyle="none", marker="o", color="black")
        canvas.arrow(self.origin, self.xdir, color="red")
        canvas.arrow(self.origin, self.zdir, color="blue")
        return self
    
    def plot_zxy(self, canvas=None):
        if canvas is None:
            canvas = CanvasZXY()
        x = self.matrix[0, 3]
        y = self.matrix[1, 3]
        z = self.matrix[2, 3]
        canvas.plot(self.origin, linestyle="none", marker="o", color="black")
        canvas.arrow(self.origin, self.xdir, color="red")
        canvas.arrow(self.origin, self.ydir, color="orange")
        canvas.arrow(self.origin, self.zdir, color="blue")
        return self

    def move_to(self, origin):
        self.matrix[0:3, 3] = origin
        return self

    def move_by(self, offset):
        self.matrix[0:3, 3] += offset
        return self

    def rotate_x(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        self.rotation = np.matmul(rot, self.rotation)
        return self

    def rotate_y(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.rotation = np.matmul(rot, self.rotation)
        return self

    def rotate_z(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.rotation = np.matmul(rot, self.rotation)
        return self

    def arc_by(self, length, angle):
        if angle == 0:
            return self.move_by([0, 0, length])
        rho = length / angle
        self.origin -= rho * self.xdir
        self.rotate_y(-angle)
        self.origin += rho * self.xdir
        return self

    def copy(self):
        return Frame(self.matrix.copy())


class BendFrame:
    def __init__(self, start, length=0, angle=0):
        self.start = start  # start frame
        self.angle = angle
        self.length = length

    @property
    def end(self):
        end = self.start.copy()
        end.arc_by(self.length, self.angle)
        return end

    def frame(self, s):
        return self.start.copy().arc_by(s, s / self.length * self.angle)

    def ref_trajectory(self, steps=11):
        return np.array(
            [self.frame(s).origin for s in np.linspace(0, self.length, steps)]
        ).T

    def trajectory(self, s,x,y):
        qq=np.zeros((3,len(s)))
        for ii,ss in enumerate(s):
            fr=self.frame(ss)
            qq[:,ii]=(fr.matrix@np.array([x[ii],y[ii],0,1]))[:3]
        return qq

    def plot_zx(self, canvas=None):
        if canvas is None:
            canvas = CanvasZX()
        self.start.plot_zx(canvas)
        self.end.plot_zx(canvas)
        canvas.plot(self.ref_trajectory(11), color="green")
        return self

    def plot_zxy(self, canvas=None):
        if canvas is None:
            canvas = CanvasZXY()
        self.start.plot_zxy(canvas)
        self.end.plot_zxy(canvas)
        canvas.plot(self.ref_trajectory(11), color="green")
        return self

    def plot_trajectory_zx(self, s,x,y, canvas=None):
        if canvas is None:
            canvas = CanvasZX()
        self.plot_zx(canvas)
        canvas.plot(self.trajectory(s,x,y), color="black")
        return self

    def plot_trajectory_zxy(self, s,x,y, canvas=None):
        if canvas is None:
            canvas = CanvasZXY()
        self.plot_zxy(canvas)
        canvas.plot(self.trajectory(s,x,y), color="black")
        
        # Set equal intervals for the three axes.
        smid, xmid, ymid = (np.max(s)+np.min(s))/2, (np.max(x)+np.min(x))/2, (np.max(y)+np.min(y))/2
        sdist, xdist, ydist = (np.max(s)-np.min(s))/2, (np.max(x)-np.min(x))/2, (np.max(y)-np.min(y))/2
        dist = max([sdist, xdist, ydist])
        canvas.ax.set_xlim(smid-dist, smid+dist)
        canvas.ax.set_ylim(xmid-dist, xmid+dist)
        canvas.ax.set_zlim(ymid-dist, ymid+dist)
        canvas.ax.set_box_aspect((1,1,1))

        return self

