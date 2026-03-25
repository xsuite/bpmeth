from numpy import *
import numpy as np


def solveconst(A, b, C, d):
    """
    Solve constrained least square problem using Lagrange multipliers
    min(|| A x - b ||_2) and constraint Cx=b
    x: n
    A: m x n
    b: m
    C: l x n
    d: l

    L = x A.T A x - 2 A.T x + b.T b + l C x
    Equivalent to:
    (A.T A   C.T )  (x)  = (A.T b)
    (C       0   )  (l)  = (d)
    
    :param A: x**i for the data points. To be multiplied with polynomial coefficients to get the value of the polynomial at the data points.
    :param b: y values for the data points.
    :param C: x**i for the constraint points. To be multiplied with polynomial coefficients to get the value of the polynomial at the constraint points.
    :param d: y values for the constraint points.
    
    """
    
    nl, nx = C.shape
    m = hstack([dot(A.T, A), C.T])
    m = vstack([m, hstack([C, zeros((nl, nl))])])
    n = hstack([dot(A.T, b), d])
    sol = linalg.solve(m, n)
    return sol[:nx]


def makeA(x, N):
    """
    Make matrix x**i, to be multiplied with polynomial coefficients to get the value of the polynomial.
    :param x: 1D array of x values for the data points.
    :param N: Maximal order.
    """

    return column_stack([x**i for i in range(N + 1)])


def makeAp(x, N):
    """
    Make matrix i x**(i-1), equivalent to derivatives when multiplied with polynomial coefficients.
    :param x: 1D array of x values for the data points.
    :param N: Maximal order.
    """
    
    return column_stack([zeros(len(x))] + [i * x ** (i - 1) for i in range(1, N + 1)])


def makeApp(x, N):
    """
    Make matrix i (i-1) x**(i-2), equivalent to second derivatives when multiplied with polynomial coefficients.
    :param x: 1D array of x values for the data points.
    :param N: Maximal order.
    """
    
    z = [zeros(len(x))] * 2
    z += [i * (i - 1) * x ** (i - 2) for i in range(2, N + 1)]
    return column_stack(z)


def poly_val(p, x):
    """ 
    Return the value of the polynomial at the given point.
    :param p: List of polynomial coefficients, where the i-th element corresponds to the coefficient of x^i.
    :param x: Point(s) at which to evaluate the polynomial, can be a scalar or a numpy array.
    """
    
    return sum([p[i] * x**i for i in range(len(p))], axis=0)


def poly_print(p, x="x", power="**", mul="*"):
    """ 
    Return the expression for the polynomial as a string.
    :param p: List of polynomial coefficients, where the i-th element corresponds to the coefficient of x^i.
    :param x: Variable name to use in the expression (default is "x").
    :param power: String to use for exponentiation (default is "**").
    :param mul: String to use for multiplication (default is "*").
    """ 
    
    res = ["%.10e" % p[0]]
    if len(p) > 1:
        res.append("%+.10e%s%s" % (p[1], mul, x))
    for i in range(2, len(p)):
        res.append("%+.10e%s%s%s%d" % (p[i], mul, x, power, i))
    return "".join(res)


def poly_fit(N, xdata, ydata, x0=[], y0=[], xp0=[], yp0=[], xpp0=[], ypp0=[]):
    """
    Fit a polynomial of order N to the data (xdata, ydata) with optional constraints on the value, first derivative, and second derivative at specified points.
    :param N: Order of the polynomial to fit.
    :param xdata: 1D array of x values for the data points.
    :param ydata: 1D array of y values for the data points, same length as xdata.
    :param x0: List of x values where the polynomial should take specific values (optional).
    :param y0: List of y values corresponding to x0 where the polynomial should take specific values (optional).
    :param xp0: List of x values where the first derivative of the polynomial should take specific values (optional).
    :param yp0: List of y values corresponding to xp0 where the first derivative of the polynomial should take specific values (optional).
    :param xpp0: List of x values where the second derivative of the polynomial should take specific values (optional).
    :param ypp0: List of y values corresponding to xpp0 where the second derivative of the polynomial should take specific values (optional).
    :return: 1D array of polynomial coefficients, where the i-th element corresponds to the coefficient of x^i.
    """
    
    A = makeA(xdata, N)  # x**i at data points
    b = ydata  # y values at data points
    
    C0 = makeA(array(x0), N)  # x0**i at constraint points
    C1 = makeAp(array(xp0), N)  # i x0**(i-1) at constraint points
    C2 = makeApp(array(xpp0), N)  # i (i-1) x0**(i-2) at constraint points
    C = vstack([C0, C1, C2])  # Constraints in x to be multiplied with polynomial coefficients
    d = hstack([y0, yp0, ypp0])  # Constraint values in y
    
    p = solveconst(A, b, C, d)
    
    return p


def fit_segment(ia,ib,x,y,yp,ypp=None,order=3):
    """
    Fit a polynomial of given order to the data in the segment defined by indices ia and ib, with constraints on 
    the value and first derivative at the endpoints. If ypp is provided, also constrain the second derivative at the endpoints. 
    :param ia: Starting index of the segment to fit.
    :param ib: Ending index of the segment to fit.
    :param x: 1D array of x values for the data points.
    :param y: 1D array of y values for the data points, same length as x.
    :param yp: 1D array of first derivative values for the data points, same length as x.
    :param ypp: 1D array of second derivative values for the data points, same length as x (optional).
    :param order: Order of the polynomial to fit (default is 3).
    """
    
    x0=[x[ia],x[ib]]  # Endpoints
    y0=[y[ia],y[ib]]
    yp0=[yp[ia],yp[ib]]
    xd=x[ia:ib+1]  # All data points in the segment
    yd=y[ia:ib+1]
    if ypp is not None:
        assert order > 5, "Second derivative fitting should be done for at least order five"
        ypp0=[ypp[ia],ypp[ib]]
        pol=poly_fit(order,xd,yd,x0,y0,x0,yp0,x0,ypp0)
    else:
        pol=poly_fit(order,xd,yd,x0,y0,x0,yp0)
    return pol


def plot_fit(ia,ib,x,y,pol,data=False, ax=None):
    """ 
    Plot the fitted polynomial over the segment defined by indices ia and ib, along with the original data points if data=True.
    """
    
    xd=x[ia:ib+1]  # All data points in the segment
    yd=y[ia:ib+1]
    yf=poly_val(pol,xd)  # Corresponding fit values of polynomial
    if ax is None:
        fig, ax = plt.subplots()
    if data:
       ax.plot(xd,yd,label='data', ls='', marker='o', color='black')
    ax.plot(xd,yf,label=poly_print(pol))
