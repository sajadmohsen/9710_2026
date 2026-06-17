import numpy as np

class Grid:
    """Class defining a one-dimensional Cartesian grid"""

    def __init__(self, lx, ly, lz, ncv):
        """Constructor
            lx .... total length of domain in x-direction [m]
            ly .... total length of domain in x-direction [m]
            lz .... total length of domain in x-direction [m]
            ncv ... number of control volumes in domain
        """
        # Store geometry
        self._ly = ly
        self._lz = lz
        
        # Store the number of control volumes
        self._ncv = ncv

        # Calculate the control volume length
        dx = lx/float(ncv)

        # Calculate the face locations
        self._xf = np.array([i*dx for i in range(ncv+1)])

        # Calculate the cell centroid locations
        self._xP = np.array([self._xf[0]] +
                            [0.5*(self._xf[i]+self._xf[i+1]) for i in range(ncv)] +
                            [self._xf[-1]])

        # Calculate face areas
        self._Af = ly*lz*np.ones(ncv+1)

        # Calculate the outer surface area for each cell
        self._Ao = (2.0*dx*ly + 2.0*dx*lz)*np.ones(ncv)

        # Calculate cell volumes
        self._vol = dx*ly*lz*np.ones(ncv)

    @property
    def ncv(self):
        """Number of control volumes in domain"""
        return self._ncv

    @property
    def xf(self):
        """Face location array"""
        return self._xf

    @property
    def xP(self):
        """Cell centroid array"""
        return self._xP

    @property
    def dx_WP(self):
        return self.xP[1:-1]-self.xP[0:-2]

    @property
    def dx_PE(self):
        return self.xP[2:]-self.xP[1:-1]

    @property
    def Af(self):
        """Face area array"""
        return self._Af

    @property
    def Aw(self):
        """West face area array"""
        return self._Af[0:-1]

    @property
    def Ae(self):
        """East face area array"""
        return self._Af[1:]

    @property
    def Ao(self):
        """Outer face area array"""
        return self._Ao

    @property
    def vol(self):
        """Cell volume array"""
        return self._vol

    @property
    def ly(self):
        return self._ly

    @property
    def lz(self):
        return self._lz

class CircularDuctGrid:
    """One-dimensional grid for a converging-diverging circular duct"""

    def __init__(self, lx, Ht, ncv):
        self._ncv = ncv
        self._lx = lx
        self._Ht = Ht

        dx = lx/float(ncv)

        self._xf = np.array([i*dx for i in range(ncv+1)])

        self._xP = np.array(
            [self._xf[0]]
            + [0.5*(self._xf[i] + self._xf[i+1]) for i in range(ncv)]
            + [self._xf[-1]]
        )

        # Radius at faces
        rf = 2.0*Ht + Ht*np.cos(2.0*np.pi*self._xf/lx)

        # Radius at cell centers
        rP_internal = 2.0*Ht + Ht*np.cos(2.0*np.pi*self._xP[1:-1]/lx)

        # Cross-sectional areas at faces
        self._Af = np.pi*rf**2

        # Volumes using cell-center area times dx
        self._vol = np.pi*rP_internal**2*dx

        # No wall friction in Problem 3, but define Ao for compatibility
        self._Ao = np.zeros(ncv)

    @property
    def ncv(self):
        return self._ncv

    @property
    def xf(self):
        return self._xf

    @property
    def xP(self):
        return self._xP

    @property
    def dx_WP(self):
        return self.xP[1:-1] - self.xP[0:-2]

    @property
    def dx_PE(self):
        return self.xP[2:] - self.xP[1:-1]

    @property
    def Af(self):
        return self._Af

    @property
    def Aw(self):
        return self._Af[0:-1]

    @property
    def Ae(self):
        return self._Af[1:]

    @property
    def Ao(self):
        return self._Ao

    @property
    def vol(self):
        return self._vol
