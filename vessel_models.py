import yaml
import numpy as np
from numpy.linalg import inv

class VesselModel:
    def __init__(self, params_url="params_real.yaml", mismatch=False):
        params_yaml = open(params_url, 'r')
        self.params = yaml.safe_load(params_yaml)
        params_yaml.close()

        if mismatch:
            for key in self.params:
                self.params[key] *= np.random.uniform(low=1.5, high=2.0)
                self.params[key] += np.random.uniform(low=0.0, high=1.0)
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
    
    def rk4_addvel(self, Sn, Un, dt):
        angle = np.pi/3
        mag = 0.05
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

        dsdt = np.zeros(6)
        dsdt[:3] = dxypsi
        dsdt[3:] = duvr
        return dsdt