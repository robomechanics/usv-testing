import math
import mmgdynamics as mmg
import mmgdynamics.calibrated_vessels as cvs
import matplotlib.pyplot as plt # Just for demostration

# Load a pre-calibrated vessel
vessel = mmg.Vessel(**cvs.kvlcc2_l64)

# Let the vessel drive with a rudder angle
# of 10° for 1000 seconds
# -------------------------------------
# Inital position
pos = [0,0] # x,y [m]

# Initial heading
psi = 0 # [rad]

# Random initial values (replace these with yours)
uvr = [3.85, 0, 0] # u,v,r [m/s, m/s, rad/s]

positions = []
for _ in range(1000):
    uvr, eta = mmg.pstep(
        X           = uvr,
        pos         = pos,
        vessel      = vessel,
        dT          = 1,    # 1 second
        nps         = 4,    # 4 revs per second
        delta       = 10 * (math.pi / 180), # Convert to radians
        psi         = psi,  # Heading
        water_depth = None, # No water depth
        fl_psi      = None, # No current angle
        fl_vel      = None, # No current velocity
        w_vel       = None, # No wind velocity
        beta_w      = None  # No wind angle
    )
    x,y,psi = eta # Unpack new position and heading
    positions.append([x,y]) # Store the new position
    pos = [x,y] # Update the position
    
# Quick plot of the trajectory
ps = list(zip(*positions))
plt.plot(ps[0], ps[1])
plt.show()