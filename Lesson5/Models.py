import numpy as np
class DiffusionModel:
    """Class defining a diffusion model"""

    def __init__(self, grid, phi, gamma, west_bc, east_bc):
        """Constructor"""
        self._grid = grid
        self._phi = phi
        self._gamma = gamma
        self._west_bc = west_bc
        self._east_bc = east_bc

    def add(self, coeffs):
        """Function to add diffusion terms to coefficient arrays"""

        # Calculate the west and east face diffusion flux terms for each face
        flux_w = - self._gamma*self._grid.Aw*(self._phi[1:-1]-self._phi[0:-2])/self._grid.dx_WP
        flux_e = - self._gamma*self._grid.Ae*(self._phi[2:]-self._phi[1:-1])/self._grid.dx_PE

        # Calculate the linearization coefficients
        coeffW = - self._gamma*self._grid.Aw/self._grid.dx_WP
        coeffE = - self._gamma*self._grid.Ae/self._grid.dx_PE
        coeffP = - coeffW - coeffE

        # Modify the linearization coefficients on the boundaries
        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        # Zero the boundary coefficients that are not used
        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        # Calculate the net flux from each cell
        flux = flux_e - flux_w

        # Add to coefficient arrays
        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)

        # Return the modified coefficient array
        return coeffs
class FirstOrderTransientModel:
    """Class defining a first-order implicit transient model"""

    def __init__(self, grid, phi, phiold, rho, const, dt):
        self._grid = grid
        self._phi = phi
        self._phiold = phiold
        self._rho = rho
        self._const = const
        self._dt = dt

    def add(self, coeffs):
        transient = (
            self._rho*self._const*self._grid.vol
            *(self._phi[1:-1] - self._phiold[1:-1])/self._dt
        )

        coeff = self._rho*self._const*self._grid.vol/self._dt

        coeffs.accumulate_aP(coeff)
        coeffs.accumulate_rP(transient)

        return coeffs
class UpwindAdvectionModel:
    """Class defining an upwind advection model"""

    def __init__(self, grid, phi, Uhe, rho, const, west_bc, east_bc):
        self._grid = grid
        self._phi = phi
        self._Uhe = Uhe
        self._rho = rho
        self._const = const
        self._west_bc = west_bc
        self._east_bc = east_bc
        self._alphae = np.zeros(self._grid.ncv + 1)

    def add(self, coeffs):
        self._alphae[:] = np.where(self._Uhe >= 0, 1, -1)

        phie = (
            (1 + self._alphae)/2*self._phi[0:-1]
            + (1 - self._alphae)/2*self._phi[1:]
        )

        mdote = self._rho*self._Uhe*self._grid.Af

        flux_w = self._const*mdote[:-1]*phie[:-1]
        flux_e = self._const*mdote[1:]*phie[1:]

        imbalance = (
            - self._const*mdote[1:]*self._phi[1:-1]
            + self._const*mdote[:-1]*self._phi[1:-1]
        )

        coeffW = -self._const*mdote[:-1]*(1 + self._alphae[:-1])/2
        coeffE = self._const*mdote[1:]*(1 - self._alphae[1:])/2
        coeffP = -coeffW - coeffE

        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        flux = flux_e - flux_w

        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)
        coeffs.accumulate_rP(imbalance)

        return coeffs

class CentralAdvectionModel:
    """Class defining a CDS advection model with UDS linearization"""

    def __init__(self, grid, phi, Uhe, rho, const, west_bc, east_bc):
        self._grid = grid
        self._phi = phi
        self._Uhe = Uhe
        self._rho = rho
        self._const = const
        self._west_bc = west_bc
        self._east_bc = east_bc
        self._alphae = np.zeros(self._grid.ncv + 1)

    def add(self, coeffs):
        self._alphae[:] = np.where(self._Uhe >= 0, 1, -1)

        # UDS face values for linearization
        phi_uds = (
            (1 + self._alphae)/2*self._phi[0:-1]
            + (1 - self._alphae)/2*self._phi[1:]
        )

        # CDS face values for residual
        phi_cds = 0.5*(self._phi[0:-1] + self._phi[1:])

        mdote = self._rho*self._Uhe*self._grid.Af

        flux_w = self._const*mdote[:-1]*phi_cds[:-1]
        flux_e = self._const*mdote[1:]*phi_cds[1:]

        imbalance = (
            - self._const*mdote[1:]*self._phi[1:-1]
            + self._const*mdote[:-1]*self._phi[1:-1]
        )

        coeffW = -self._const*mdote[:-1]*(1 + self._alphae[:-1])/2
        coeffE = self._const*mdote[1:]*(1 - self._alphae[1:])/2
        coeffP = -coeffW - coeffE

        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        flux = flux_e - flux_w

        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)
        coeffs.accumulate_rP(imbalance)

        return coeffs

class QuickAdvectionModel:
    """Class defining a QUICK advection model with UDS linearization"""

    def __init__(self, grid, phi, Uhe, rho, const, west_bc, east_bc):
        self._grid = grid
        self._phi = phi
        self._Uhe = Uhe
        self._rho = rho
        self._const = const
        self._west_bc = west_bc
        self._east_bc = east_bc
        self._alphae = np.zeros(self._grid.ncv + 1)

    def add(self, coeffs):
        self._alphae[:] = np.where(self._Uhe >= 0, 1, -1)

        # UDS face values for linearization
        phi_uds = (
            (1 + self._alphae)/2*self._phi[0:-1]
            + (1 - self._alphae)/2*self._phi[1:]
        )

        # QUICK face values for residual
        phi_quick = np.copy(phi_uds)

        for i in range(1, self._grid.ncv):
            if self._Uhe[i] >= 0:
                phi_quick[i] = (
                    3.0*self._phi[i+1]
                    + 6.0*self._phi[i]
                    - self._phi[i-1]
                )/8.0
            else:
                phi_quick[i] = (
                    3.0*self._phi[i]
                    + 6.0*self._phi[i+1]
                    - self._phi[i+2]
                )/8.0

        mdote = self._rho*self._Uhe*self._grid.Af

        flux_w = self._const*mdote[:-1]*phi_quick[:-1]
        flux_e = self._const*mdote[1:]*phi_quick[1:]

        imbalance = (
            - self._const*mdote[1:]*self._phi[1:-1]
            + self._const*mdote[:-1]*self._phi[1:-1]
        )

        coeffW = -self._const*mdote[:-1]*(1 + self._alphae[:-1])/2
        coeffE = self._const*mdote[1:]*(1 - self._alphae[1:])/2
        coeffP = -coeffW - coeffE

        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        flux = flux_e - flux_w

        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(flux)
        coeffs.accumulate_rP(imbalance)

        return coeffs
        
class PressureForceModel:
    """Class defining the pressure-gradient term in the momentum equation"""

    def __init__(self, grid, P, west_bc, east_bc):
        self._grid = grid
        self._P = P
        self._west_bc = west_bc
        self._east_bc = east_bc

    def add(self, coeffs):
        gradPw = (self._P[1:-1] - self._P[0:-2])/self._grid.dx_WP
        gradPe = (self._P[2:] - self._P[1:-1])/self._grid.dx_PE

        force = 0.5*(gradPw + gradPe)*self._grid.vol

        coeffW = -0.5*self._grid.vol/self._grid.dx_WP
        coeffE = 0.5*self._grid.vol/self._grid.dx_PE
        coeffP = -coeffW - coeffE

        coeffP[0] += coeffW[0]*self._west_bc.coeff()
        coeffP[-1] += coeffE[-1]*self._east_bc.coeff()

        coeffW[0] = 0.0
        coeffE[-1] = 0.0

        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_aW(coeffW)
        coeffs.accumulate_aE(coeffE)
        coeffs.accumulate_rP(force)

        return coeffs

class WallFrictionModel:
    """Class defining a turbulent wall-friction source term"""

    def __init__(self, grid, U, rho, mu):
        self._grid = grid
        self._U = U
        self._rho = rho
        self._mu = mu

    def add(self, coeffs):

        # Hydraulic diameter
        A = self._grid.ly * self._grid.lz
        Po = 2.0 * (self._grid.ly + self._grid.lz)
        Dh = 4.0 * A / Po

        # Cell-center velocity magnitude
        Uc = np.abs(self._U[1:-1])

        # Reynolds number
        Re = self._rho * Dh * np.maximum(Uc, 1e-12) / self._mu

        # Friction coefficient
        Cf = (1.58*np.log(Re) - 3.28)**(-2)

        # Linearization of U² term
        coeffP = self._rho * Cf * Uc * self._grid.Ao

        source = 0.5 * self._rho * Cf * Uc * self._U[1:-1] * self._grid.Ao

        coeffs.accumulate_aP(coeffP)
        coeffs.accumulate_rP(source)

        return coeffs
        
class AdvectingVelocityModel:
    """Class defining the Rhie-Chow advecting velocity"""

    def __init__(self, grid, dhat, Uhe, P, U, coeffs):
        self._grid = grid
        self._dhat = dhat
        self._Uhe = Uhe
        self._P = P
        self._U = U
        self._coeffs = coeffs

    def update(self):
        gradPw = (self._P[1:-1] - self._P[0:-2])/self._grid.dx_WP
        gradPe = (self._P[2:] - self._P[1:-1])/self._grid.dx_PE

        gradP = 0.5*(gradPw + gradPe)

        Ve = 0.5*(self._grid.vol[0:-1] + self._grid.vol[1:])
        ae = 0.5*(self._coeffs.aP[0:-1] + self._coeffs.aP[1:])

        self._dhat[1:-1] = Ve/ae

        self._Uhe[0] = self._U[0]

        self._Uhe[1:-1] = (
            0.5*(self._U[1:-2] + self._U[2:-1])
            - self._dhat[1:-1]*
            (gradPe[:-1] - 0.5*(gradP[:-1] + gradP[1:]))
        )

        self._Uhe[-1] = self._U[-1]
class MassConservationEquation:
    """Class defining the coupled mass conservation equation"""

    def __init__(self, grid, U, P, dhat, Uhe, rho,
                 P_west_bc, P_east_bc, U_west_bc, U_east_bc):

        self._grid = grid
        self._U = U
        self._P = P
        self._dhat = dhat
        self._Uhe = Uhe
        self._rho = rho
        self._P_west_bc = P_west_bc
        self._P_east_bc = P_east_bc
        self._U_west_bc = U_west_bc
        self._U_east_bc = U_east_bc

    def add(self, PP_coeffs, PU_coeffs):
        imbalance = (
            self._rho*self._grid.Ae*self._Uhe[1:]
            - self._rho*self._grid.Aw*self._Uhe[:-1]
        )

        PP_coeffW = np.concatenate((
            np.array([0.0]),
            -self._rho*self._grid.Aw[1:]*
            self._dhat[1:-1]/self._grid.dx_WP[1:]
        ))

        PP_coeffE = np.concatenate((
            -self._rho*self._grid.Ae[:-1]*
            self._dhat[1:-1]/self._grid.dx_PE[:-1],
            np.array([0.0])
        ))

        PP_coeffP = -PP_coeffW - PP_coeffE

        PU_coeffW = np.concatenate((
            np.array([-self._rho*self._grid.Aw[0]]),
            -0.5*self._rho*self._grid.Aw[1:]
        ))

        PU_coeffE = np.concatenate((
            0.5*self._rho*self._grid.Ae[:-1],
            np.array([self._rho*self._grid.Ae[-1]])
        ))

        PU_coeffP = (
            np.concatenate((np.array([0.0]), PU_coeffW[1:]))
            + np.concatenate((PU_coeffE[:-1], np.array([0.0])))
        )

        PU_coeffP[0] += PU_coeffW[0]*self._U_west_bc.coeff()
        PU_coeffP[-1] += PU_coeffE[-1]*self._U_east_bc.coeff()

        PU_coeffW[0] = 0.0
        PU_coeffE[-1] = 0.0

        PP_coeffs.accumulate_aP(PP_coeffP)
        PP_coeffs.accumulate_aW(PP_coeffW)
        PP_coeffs.accumulate_aE(PP_coeffE)
        PP_coeffs.accumulate_rP(imbalance)

        PU_coeffs.accumulate_aP(PU_coeffP)
        PU_coeffs.accumulate_aW(PU_coeffW)
        PU_coeffs.accumulate_aE(PU_coeffE)

        return PP_coeffs, PU_coeffs
