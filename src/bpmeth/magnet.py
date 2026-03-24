from .fast_hamilton_solver import PolySegment
import numpy as np
import sympy as sp
import warnings
import matplotlib.pyplot as plt
import scipy as sc
import math


def pol_to_comp(pols):
    """
    :param pols: Polynomial coefficients from the fit for this segment, 
        first index is the multipole order (only normal components) b1, b2, b3..., 
        second index is the order of the term in the cubic polynomial (zero = constant)
    :return: comp, the input needed for the segment tracker
    """
    
    # comp has bs, b1, a1, b2, a2, ... as first index, and the order of the term in the cubic polynomial as second index (zero = constant)
    comp = np.zeros((9, 4), dtype=np.float64) 
     
    for mult in range(len(pols)):
        comp[2*mult+1] = pols[mult]
    
    return comp

 
class RK4_Magnet:
    isthick=True
    def __init__(self, segments, all_pols, z_edge, rho, ds=0.01):
        """
        Magnet element that can be used with Xsuite, integrates the magnet with given segments and polynomials.
        :param segments: List of (a,b) tuples defining the segments along z, entrance half of the magnet.
        :param all_pols: For each multipole, list of lists of polynomials for each segment, entrance half 
        of the magnet. This has shape (n_multipoles, n_segments, order+1)
        :param z_edge: Longitudinal position of the edge of the magnet, to determine curvature.
        :param rho: Bending radius of the magnet.
        :param ds: Step size for the integration, best to do convergence study for specific magnet.
        """
        
        self.segments = segments
        self.all_pols = all_pols
        
        self.z_edge = z_edge
        self.rho = rho
                   
        self.length = segments[-1][1] - segments[0][0]
        self.build_tracker()
        self.ds = ds
        
    def build_tracker(self):
        self.polysegments = []
        for i, segment in enumerate(self.segments):
            comp = pol_to_comp(self.all_pols[:, i])
            
            if segment[0] < -self.z_edge:
                h = 0.  # Magic, this has to be a float otherwise it is extremely slow
                s_start = segment[0]
                s_end = np.min([segment[1], -self.z_edge])
                self.polysegments.append(PolySegment(length=s_end - s_start, h=h, comp=comp, s_start = s_start))
            if segment[1] > -self.z_edge and segment[0] < self.z_edge:
                h = 1/self.rho
                s_start = np.max([segment[0], -self.z_edge])
                s_end = np.min([segment[1], self.z_edge])
                self.polysegments.append(PolySegment(length=s_end - s_start, h=h, comp=comp, s_start = s_start))
            if segment[1] > self.z_edge:
                h = 0.
                s_start = np.max([segment[0], self.z_edge])
                s_end = segment[1]
                self.polysegments.append(PolySegment(length=s_end - s_start, h=h, comp=comp, s_start = s_start))
        
    def rescale(self, scalefactor):
        self.all_pols = np.array([[pol*scalefactor for pol in pols] for pols in self.all_pols])
        self.build_tracker()
        return self
        
    def copy(self):
        return RK4_Magnet(self.segments, self.all_pols, self.z_edge, self.rho)
    
    def track(self, particles):
        for i, polysegment in enumerate(self.polysegments):
            polysegment.track(particles, ds=self.ds)
            
    def track_step_by_step(self, particles):
        out = []
        for polysegment in self.polysegments:
            seg_out = polysegment.track_step_by_step(particles, ds=self.ds)
            out.extend(seg_out)
        return out
    
    def track_segment_by_segment(self, particles):
        out = [particles.copy()]
        for polysegment in self.polysegments:
            polysegment.track(particles, ds=self.ds)
            out.append(particles.copy())
        return out
        
