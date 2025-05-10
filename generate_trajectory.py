import math
import mmgdynamics as mmg
import mmgdynamics.calibrated_vessels as cvs
import matplotlib.pyplot as plt
import numpy as np

class ReferenceTrajectory:
    def __init__(self, vessel, N, dT):
        self.vessel = vessel
        self.N = N
        self.dT = dT
        
    def generate_reference(self, pos_init: np.ndarray, psi_init: float, uvr_init: np.ndarray, 
                           nps_arr: np.ndarray, delta_arr: np.ndarray, plotting=False):
        pos = pos_init
        psi = psi_init
        uvr = uvr_init
        state_init = np.concatenate((pos, np.array([psi]), uvr))
        states = [state_init]

        for i in range(self.N):
            nps = nps_arr[i]
            delta = delta_arr[i]

            uvr, eta = mmg.pstep(
                X           = uvr,
                pos         = pos,
                vessel      = self.vessel,
                dT          = self.dT,    # 1 second
                nps         = nps,    # 4 revs per second
                delta       = delta * (math.pi / 180), # Convert to radians
                psi         = psi,  # Heading
                water_depth = None, # No water depth
                fl_psi      = None, # No current angle
                fl_vel      = None, # No current velocity
                w_vel       = None, # No wind velocity
                beta_w      = None  # No wind angle
            )
            x, y, psi = eta # Unpack new position and heading
            state = np.concatenate((eta, uvr))
            states.append(state)
            pos = [x, y]

        # Quick plot of the trajectory
        if plotting:
            ps = list(zip(*states))
            plt.plot(ps[0], ps[1])
            plt.show()

        return states

if __name__ == "__main__":
    vessel = mmg.Vessel(**cvs.kvlcc2_l64)
    # Inital position
    pos = np.array([0,0]) # x,y [m]

    # Initial heading
    psi = 0.0 # [rad]

    # Random initial values (replace these with yours)
    uvr = np.array([3.85, 0, 0]) # u,v,r [m/s, m/s, rad/s]

    # Number of timesteps
    N = 1000
    dT = 1 # in seconds

    # Generate reference inputs
    nps_arr = np.ones(N) * 4
    delta_arr = np.ones(N) * 10
    delta_arr[N-500:N] = -10

    generator = ReferenceTrajectory(vessel=vessel, N=N, dT=dT)
    generator.generate_reference(pos_init=pos, psi_init=psi, uvr_init=uvr, nps_arr=nps_arr, delta_arr=delta_arr, plotting=True)