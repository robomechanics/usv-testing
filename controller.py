from vessel_models import VesselModel
import numpy as np
import cyipopt as ipopt
import os

class TrajOpt:
    def __init__(self, model: VesselModel, N, ns, nu, dt, Qvec, Rvec, Qfvec, s0, sf, obstacles):
        self.model = model
        self.N = N
        self.ns = ns
        self.nu = nu
        self.dt = dt

        self.Qvec = Qvec
        self.Rvec = Rvec
        self.Qfvec = Qfvec
        self.Q = np.diag(Qvec)
        self.R = np.diag(Rvec)
        self.Qf = np.diag(Qfvec)

        self.s0 = s0
        self.sf = sf

        # Control clamps
        self.u_min = -10
        self.u_max = 10

        # Obstacles
        self.obs = obstacles

        self.Z0 = self.make_initial_guess()
        assert len(self.Z0) == N*ns + (N-1)*nu, "Length of initial guess does not match length calculated from given N, nx, and nu"

    def make_initial_guess(self):
        states = np.zeros((self.N,self.ns))
        states[0] = self.s0

        inputs = np.zeros((self.N-1, self.nu))

        states_flat = states.flatten()
        inputs_flat = inputs.flatten()
        guess = np.concatenate((states_flat, inputs_flat))
        return guess
    
    def flat2vec(self, Z):
        states_flat = Z[:self.N*self.ns]
        inputs_flat = Z[self.N*self.ns:]
        states = states_flat.reshape((self.N, self.ns))
        inputs = inputs_flat.reshape((self.N-1, self.nu))
        return states, inputs
    
    def LQRcost_fast(self, Z):
        states, inputs = self.flat2vec(Z)
        S = states[:-1]

        cost = 0
        cost += np.sum(0.5 * np.vecdot((S-self.sf), np.multiply(self.Qvec, S-self.sf)))
        cost += np.sum(0.5 * np.vecdot((states[-1]-self.sf), np.multiply(self.Qfvec, states[-1]-self.sf)))
        #print(f"cost from states: {cost}")
        cost += np.sum(0.5 * np.vecdot(inputs, np.multiply(self.Rvec, inputs)))
        #print(f"cost with inputs: {cost}")

        return cost
    
    def equality_constraints_fast(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints = np.zeros((self.N-1+2, self.ns))

        # Initial and terminal constraints
        initial_constraint = states[0] - self.s0
        terminal_constraint = states[-1] - self.sf
        constraints[-1] = initial_constraint
        constraints[-2] = terminal_constraint

        # Dynamics constraints
        for i in range(self.N-1):
            sn = states[i]
            snp1 = states[i+1]
            un = inputs[i]
            #dyn_snp1 = self.model.euler(sn, un, self.dt)
            #dynamics_constraint = snp1 - dyn_snp1
            dynamics_constraint = self.model.hermite_simpson(sn, snp1, un, self.dt)
            constraints[i] = dynamics_constraint

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return constraints.flatten()
    
    def inequality_constraints_fast(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints_min_controls = inputs - self.u_min
        constraints_max_controls = -inputs + self.u_max
        constraints = np.concatenate((constraints_min_controls.flatten(), constraints_max_controls.flatten()))
                                      
        X = states[:,0]
        Y = states[:,1]
        for obstacle in self.obs:
            obs_x, obs_y, obs_r = obstacle
            oc = (X-obs_x)**2 + (Y-obs_y)**2 - obs_r**2
            constraints = np.concatenate((constraints, oc.flatten()))

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def exec_trajopt(self):
        constraints = [
            {"type": "eq", "fun": self.equality_constraints_fast},
            {"type": "ineq", "fun": self.inequality_constraints_fast}
        ]

        self.sol = ipopt.minimize_ipopt(
            fun=self.LQRcost_fast,
            x0=self.Z0,
            constraints=constraints,
            tol=5e-5,
            options={"disp": 5,
                     'linear_solver': 'ma57',
                     'hsllib': f"{os.environ['CONDA_PREFIX']}/lib/x86_64-linux-gnu/libcoinhsl.so"}
        )

class MPC:
    def __init__(self, model: VesselModel, N, ns, nu, dt, Qvec, Rvec, Qfvec, s0, sf, sref, obstacles, radii):
        self.model = model
        self.N = N # this is the MPC window size, not the N of the entire trajectory
        self.ns = ns
        self.nu = nu
        self.dt = dt

        self.Qvec = Qvec
        self.Rvec = Rvec
        self.Qfvec = Qfvec
        self.Q = np.diag(Qvec)
        self.R = np.diag(Rvec)
        self.Qf = np.diag(Qfvec)

        self.s0 = s0
        self.sf = sf
        self.sref = sref # reference trajectory MPC is following

        # Control clamps
        self.u_min = -10
        self.u_max = 10

        # Obstacle states
        self.obs = obstacles
        self.radii = radii

        self.Z0 = self.make_initial_guess()
        assert len(self.Z0) == N*ns + (N-1)*nu, "Length of initial guess does not match length calculated from given N, nx, and nu"

    def make_initial_guess(self):
        states = np.zeros((self.N,self.ns))
        states[0] = self.s0

        inputs = np.zeros((self.N-1, self.nu))

        states_flat = states.flatten()
        inputs_flat = inputs.flatten()
        guess = np.concatenate((states_flat, inputs_flat))
        return guess
    
    def flat2vec(self, Z):
        states_flat = Z[:self.N*self.ns]
        inputs_flat = Z[self.N*self.ns:]
        states = states_flat.reshape((self.N, self.ns))
        inputs = inputs_flat.reshape((self.N-1, self.nu))
        return states, inputs
    
    def LQRcost_fast(self, Z):
        states, inputs = self.flat2vec(Z)
        S = states[:-1]

        cost = 0
        cost += np.sum(0.5 * np.vecdot((S-self.sref[:-1]), np.multiply(self.Qvec, S-self.sref[:-1])))
        cost += np.sum(0.5 * np.vecdot((states[-1]-self.sf), np.multiply(self.Qfvec, states[-1]-self.sf)))
        #print(f"cost from states: {cost}")
        cost += np.sum(0.5 * np.vecdot(inputs, np.multiply(self.Rvec, inputs)))
        #print(f"cost with inputs: {cost}")

        return cost
    
    def LQRcost_dist(self, Z):
        states, inputs = self.flat2vec(Z)
        cost = 0
        cost += np.sum(0.5 * np.vecdot((states-self.sf), np.multiply(self.Qvec, states-self.sf)))
        cost += np.sum(0.5 * np.vecdot(inputs, np.multiply(self.Rvec, inputs)))

        # "Soft" Barrier function for obstacles
        """
        for i in range(len(self.radii)):
            obstacle = self.obs[i]
            radius = self.radii[i]

            xys = states-obstacle
            rs = np.sum(xys[:, 0:2] ** 2, axis=1)
            b = rs - (radius**2)
            bcost = np.exp(-100*b)
            cost += np.sum(bcost)
        """

        return cost
    
    def equality_constraints_fast(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints = np.zeros((self.N-1+2, self.ns))

        # Initial and terminal constraints
        initial_constraint = states[0] - self.s0
        terminal_constraint = states[-1] - self.sf
        constraints[-1] = initial_constraint
        constraints[-2] = terminal_constraint

        # Dynamics constraints
        for i in range(self.N-1):
            sn = states[i]
            snp1 = states[i+1]
            un = inputs[i]
            #dyn_snp1 = self.model.euler(sn, un, self.dt)
            #dynamics_constraint = snp1 - dyn_snp1
            dynamics_constraint = self.model.hermite_simpson(sn, snp1, un, self.dt)
            constraints[i] = dynamics_constraint

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return constraints.flatten()
    
    def inequality_constraints_fast(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints_min_controls = inputs - self.u_min
        constraints_max_controls = -inputs + self.u_max
        
        constraints = np.concatenate((constraints_min_controls.flatten(), constraints_max_controls.flatten()))
        #print(f"constraints = {constraints}")

        # Obstacle constraints
        """
        for i in range(len(self.radii)):
            obstacle = self.obs[i]
            radius = self.radii[i]

            xys = states-obstacle
            rs = np.sum(xys[:, 0:2] ** 2, axis=1)
            b = rs - (radius**2)
            #print(f"b = {b}")
            constraints = np.concatenate((constraints, b.flatten()))
        """

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def exec_MPC(self):
        constraints = [
            {"type": "eq", "fun": self.equality_constraints_fast},
            {"type": "ineq", "fun": self.inequality_constraints_fast}
        ]

        self.sol = ipopt.minimize_ipopt(
            #fun=self.LQRcost_fast,
            fun=self.LQRcost_dist,
            x0=self.Z0,
            constraints=constraints,
            tol=1e-4, # default 1e-3
            options={"disp": 5,
                     'linear_solver': 'ma57',
                     'hsllib': f"{os.environ['CONDA_PREFIX']}/lib/x86_64-linux-gnu/libcoinhsl.so"}
        )