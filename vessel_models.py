import yaml
import numpy as np
from numpy.linalg import inv

class VesselModel:
    def __init__(self, params_url="params_real.yaml", mismatch=False):
        params_yaml = open(params_url, 'r')
        self.params = yaml.safe_load(params_yaml)
        params_yaml.close()

        if mismatch:
            param_keys = list(self.params.keys())
            for i in range(13):
                key = param_keys[i]
                self.params[key] *= 2.0
                self.params[key] += 1.0
            for i in range(13, len(param_keys)):
                key = param_keys[i]
                self.params[key] *= 1.25
                self.params[key] += 0.0
            print(self.params)

        keys = ['m', 'x_g', 'I_z', 'X_du', 'Y_dv', 'Y_dr', 'N_dv', 'N_dr']
        self.m, self.x_g, self.I_z, self.X_du, self.Y_dv, self.Y_dr, self.N_dv, self.N_dr = self.retrieve_params(keys)

        keys = ['X_u', 'X_uu', 'Y_v', 'Y_vv', 'Y_r', 'N_v', 'N_r', 'N_rr']
        self.X_u, self.X_uu, self.Y_v, self.Y_vv, self.Y_r, self.N_v, self.N_r, self.N_rr = self.retrieve_params(keys)

        self.M = np.zeros((3,3))
        self.M[0,0] = self.m - self.X_du
        self.M[1,1] = self.m - self.Y_dv
        self.M[1,2] = self.m*self.x_g - self.Y_dr
        self.M[2,1] = self.m*self.x_g - self.N_dv
        self.M[2,2] = self.I_z - self.N_dr
        #print(M)

        self.Minv = inv(self.M)

    def rk4(self, Sn, Un, dt):
        k1 = self.dynamics(Sn, Un)
        k2 = self.dynamics(Sn + dt*k1/2, Un)
        k3 = self.dynamics(Sn + dt*k2/2, Un)
        k4 = self.dynamics(Sn + dt*k3, Un)
        Snp1 = Sn + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return Snp1
    
    """
    Extended dynamics RK4 function. Assumes input state is augmented with hydrodynamic coefficients
    """
    def extended_rk4(self, Sn, Un, Cn, dt):
        k1 = self.extended_dynamics(Sn, Un, Cn)
        k2 = self.extended_dynamics(Sn + dt*k1/2, Un, Cn)
        k3 = self.extended_dynamics(Sn + dt*k2/2, Un, Cn)
        k4 = self.extended_dynamics(Sn + dt*k3, Un, Cn)
        Snp1 = Sn + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return Snp1
    
    def rk4_addvel(self, Sn, Un, dt):
        angle = np.pi/3
        mag = 0.1
        added_vel = np.array([mag*np.cos(angle), mag*np.sin(angle), 0, 0, 0, 0])

        k1 = self.dynamics(Sn, Un) + added_vel
        k2 = self.dynamics(Sn + dt*k1/2, Un) + added_vel
        k3 = self.dynamics(Sn + dt*k2/2, Un) + added_vel
        k4 = self.dynamics(Sn + dt*k3, Un) + added_vel
        Snp1 = Sn + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return Snp1
    
    def euler(self, Sn, Un, dt):
        Snp1 = Sn + dt*self.dynamics(Sn, Un)
        return Snp1
    
    def hermite_simpson(self, S1, S2, Un, dt):
        ds1 = self.dynamics(S1, Un)
        ds2 = self.dynamics(S2, Un)

        midpoint = 0.5 * (S1 + S2) + dt/8.0 * (ds1 - ds2)
        dmid = self.dynamics(midpoint, Un)

        f = S1 + dt/6.0 * (ds1 + 4.0 * dmid + ds2) - S2
        return f
    
    def extended_hermite_simpson(self, S1, S2, Un, Cn, dt):
        ds1 = self.extended_dynamics(S1, Un, Cn)
        ds2 = self.extended_dynamics(S2, Un, Cn)

        midpoint = 0.5 * (S1 + S2) + dt/8.0 * (ds1 - ds2)
        dmid = self.extended_dynamics(midpoint, Un, Cn)

        f = S1 + dt/6.0 * (ds1 + 4.0 * dmid + ds2) - S2
        return f
    
    def retrieve_params(self, keys):
        values = [self.params[key] for key in keys]
        return values
    
    def dynamics(self, Sn: np.ndarray, Un: np.ndarray):
        # state s in form [x y psi u v r]
        x, y, psi, u, v, r = Sn

        # Calculate change in NED position
        R = np.eye(3)
        R[0,0] = np.cos(psi)
        R[0,1] = -np.sin(psi)
        R[1,0] = np.sin(psi)
        R[1,1] = np.cos(psi)
        #print(R)

        uvr = np.array([u,v,r])
        dxypsi = R @ uvr

        # Calculate change in  body velocity
        C = np.zeros((3,3))
        C[0,2] = -(self.m-self.Y_dv)*v - (self.m*self.x_g-self.Y_dr)*r
        C[1,2] = (self.m-self.X_du)*u
        C[2,0] = (self.m-self.Y_dv)*v + (self.m*self.x_g-self.Y_dr)*r
        C[2,1] = -(self.m-self.X_du)*u
        #print(C)

        D = np.zeros((3,3))
        D[0,0] = -self.X_u - self.X_uu*np.abs(u)
        D[1,1] = -self.Y_v - self.Y_vv*np.abs(v)
        D[1,2] = -self.Y_r
        D[2,1] = -self.N_v
        D[2,2] = -self.N_r - self.N_rr*np.abs(r)
        #print(D)

        duvr = self.Minv@Un - self.Minv@(C+D)@uvr

        dsdt = np.zeros(len(Sn))
        dsdt[:3] = dxypsi
        dsdt[3:] = duvr
        return dsdt
    
    """
    Extended dynamics function. Assumes input state is augmented with hydrodynamic coefficients 
    """
    def extended_dynamics(self, Sn: np.ndarray, Un: np.ndarray, Z: np.ndarray):
        # state s in form [x y psi u v r]
        # Hydrodynamic coeff Cn in form [X_u X_uu Y_v Y_vv Y_r N_v N_r N_rr X_du Y_dv Y_dr N_dv N_dr m I_z x_g] + [G...] of fudge factors
        x, y, psi, u, v, r = Sn

        #print(Z)
        Cn = Z[:16]
        Gn = Z[16:]
        X_u, X_uu, Y_v, Y_vv, Y_r, N_v, N_r, N_rr, X_du, Y_dv, Y_dr, N_dv, N_dr, m, I_z, x_g = Cn
        G_u, G_v, G_r, G_uu, G_vv, G_rr = Gn
        #print(f"m = {m}")

        # Calculate mass matrix
        M = np.zeros((3,3))
        M[0,0] = m - X_du
        M[1,1] = m - Y_dv
        M[1,2] = m*x_g - Y_dr
        M[2,1] = m*x_g - N_dv
        M[2,2] = I_z - N_dr
        #print(M)

        Minv = inv(M)

        # Calculate change in NED position
        R = np.eye(3)
        R[0,0] = np.cos(psi)
        R[0,1] = -np.sin(psi)
        R[1,0] = np.sin(psi)
        R[1,1] = np.cos(psi)
        #print(R)

        uvr = np.array([u,v,r])
        dxypsi = R @ uvr

        # Calculate change in  body velocity
        C = np.zeros((3,3))
        C[0,2] = -(m-Y_dv)*v - (m*x_g-Y_dr)*r
        C[1,2] = (m-X_du)*u
        C[2,0] = (m-Y_dv)*v + (m*x_g-Y_dr)*r
        C[2,1] = -(m-X_du)*u
        #print(C)

        D = np.zeros((3,3))
        D[0,0] = -X_u - X_uu*np.abs(u)
        D[1,1] = -Y_v - Y_vv*np.abs(v)
        D[1,2] = -Y_r
        D[2,1] = -N_v
        D[2,2] = -N_r - N_rr*np.abs(r)
        #print(D)

        duvr = Minv@Un - Minv@(C+D)@uvr
        duvr += np.array([G_u, G_v, G_r])
        duvr += np.array([G_uu**2, G_vv**2, G_rr**2])

        dsdt = np.zeros(len(Sn))
        dsdt[:3] = dxypsi
        dsdt[3:] = duvr
        return dsdt