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

        #constraints_min_controls = inputs - self.u_min
        constraints_min_controls_u = inputs[:, 0] - self.u_min * 0.2
        constraints_min_controls_v = inputs[:, 1] - self.u_min * 0.5
        constraints_min_controls_r = inputs[:, 2] - self.u_min

        #constraints_max_controls = -inputs + self.u_max
        constraints_max_controls_u = -inputs[:, 0] + self.u_max
        constraints_max_controls_v = -inputs[:, 1] + self.u_max * 0.5
        constraints_max_controls_r = -inputs[:, 2] + self.u_max
        
        #constraints = np.concatenate((constraints_min_controls.flatten(), constraints_max_controls.flatten()))
        constraints = np.concatenate((constraints_min_controls_u.flatten(), constraints_min_controls_v.flatten(), constraints_min_controls_r.flatten(),
                                      constraints_max_controls_u.flatten(), constraints_max_controls_v.flatten(), constraints_max_controls_r.flatten()))
                                      
        X = states[:,0]
        Y = states[:,1]
        for obstacle in self.obs:
            obs_x, obs_y, obs_r = obstacle
            oc = (X-obs_x)**2 + (Y-obs_y)**2 - obs_r**2
            constraints = np.concatenate((constraints, oc.flatten()))

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def exec_trajopt(self):
        hsllib_path = f"{os.environ['CONDA_PREFIX']}/lib/x86_64-linux-gnu/libcoinhsl.so" # for local machine
        #hsllib_path = f"{os.environ['CONDA_PREFIX']}/lib64/libcoinhsl.so" # for NCSA

        constraints = [
            {"type": "eq", "fun": self.equality_constraints_fast},
            {"type": "ineq", "fun": self.inequality_constraints_fast}
        ]

        self.sol = ipopt.minimize_ipopt(
            fun=self.LQRcost_fast,
            x0=self.Z0,
            constraints=constraints,
            tol=1e-3,
            options={"disp": 5,
                     'linear_solver': 'ma57',
                     'hsllib': hsllib_path}
        )

class MPC:
    def __init__(self, model: VesselModel, N, ns, nu, dt, Qvec, Rvec, Qfvec, s0, sf, sref, obstacles):
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
        self.u_min = -10 # default +-10
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
        cost += np.sum(0.5 * np.vecdot((S-self.sref[:-1]), np.multiply(self.Qvec, S-self.sref[:-1])))
        cost += np.sum(0.5 * np.vecdot((states[-1]-self.sf), np.multiply(self.Qfvec, states[-1]-self.sf)))
        #print(f"cost from states: {cost}")
        cost += np.sum(0.5 * np.vecdot(inputs, np.multiply(self.Rvec, inputs)))
        #print(f"cost with inputs: {cost}")

        return cost
    
    def LQRcost_dist(self, Z):
        states, inputs = self.flat2vec(Z)
        S = states[:-1]

        cost = 0
        cost += np.sum(0.5 * np.vecdot((S-self.sf), np.multiply(self.Qvec, S-self.sf)))
        cost += np.sum(0.5 * np.vecdot((states[-1]-self.sf), np.multiply(self.Qfvec, states[-1]-self.sf)))
        #print(f"cost from states: {cost}")

        cost += np.sum(0.5 * np.vecdot(inputs, np.multiply(self.Rvec, inputs)))
        #print(f"cost with inputs: {cost}")

        # "Soft" Barrier function for obstacles
        """
        for obstacle in self.obs:
            #print(f"obstacle = {obstacle}")
            obs_xy = obstacle[0:2]
            obs_r = obstacle[2]

            xys = states[:, 0:2]-obs_xy
            #print(f"xys = {xys}")

            rs = np.sum(xys ** 2, axis=1)
            #print(f"rs = {rs}")

            b = rs - (obs_r**2)
            #print(f"b = {b}")
            
            bcost = 100 * np.exp(-(100*b+5))
            #print(f"bcost = {bcost}")

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
    
    def extended_equality_constraints_fast(self, Z):
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
            dynamics_constraint = self.model.extended_hermite_simpson(sn, snp1, un, self.MHE_coeff, self.dt)
            constraints[i] = dynamics_constraint

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return constraints.flatten()
    
    def inequality_constraints_fast(self, Z):
        states, inputs = self.flat2vec(Z)

        #constraints_min_controls = inputs - self.u_min
        constraints_min_controls_u = inputs[:, 0] - self.u_min * 0.2
        constraints_min_controls_v = inputs[:, 1] - self.u_min * 0.5
        constraints_min_controls_r = inputs[:, 2] - self.u_min

        #constraints_max_controls = -inputs + self.u_max
        constraints_max_controls_u = -inputs[:, 0] + self.u_max
        constraints_max_controls_v = -inputs[:, 1] + self.u_max * 0.5
        constraints_max_controls_r = -inputs[:, 2] + self.u_max
        
        #constraints = np.concatenate((constraints_min_controls.flatten(), constraints_max_controls.flatten()))
        constraints = np.concatenate((constraints_min_controls_u.flatten(), constraints_min_controls_v.flatten(), constraints_min_controls_r.flatten(),
                                      constraints_max_controls_u.flatten(), constraints_max_controls_v.flatten(), constraints_max_controls_r.flatten()))
        X = states[:,0]
        Y = states[:,1]
        for obstacle in self.obs:
            obs_x, obs_y, obs_r = obstacle
            oc = (X-obs_x)**2 + (Y-obs_y)**2 - obs_r**2
            constraints = np.concatenate((constraints, oc.flatten()))

        #print(f"constraints = {constraints}")

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def exec_MPC(self, version="with_traj", MHE_coeff=None):
        print(f"Running MPC Solver Version: {version}")

        hsllib_path = f"{os.environ['CONDA_PREFIX']}/lib/x86_64-linux-gnu/libcoinhsl.so" # for local machine
        #hsllib_path = f"{os.environ['CONDA_PREFIX']}/lib64/libcoinhsl.so" # for NCSA

        constraints = [
            {"type": "eq", "fun": self.equality_constraints_fast},
            {"type": "ineq", "fun": self.inequality_constraints_fast}
        ]

        if version == "with_traj":
            print(f"Executed MPC Solver Version: with_traj")
            self.sol = ipopt.minimize_ipopt(
                fun=self.LQRcost_fast,
                x0=self.Z0,
                constraints=constraints,
                tol=1e-3, # default 1e-3
                options={"disp": 5,
                         'maxiter': 400,
                         'linear_solver': 'ma57',
                         'hsllib': hsllib_path}
            )
        
        if version == "without_traj":
            print(f"Executed MPC Solver Version: without_traj")
            self.sol = ipopt.minimize_ipopt(
                fun=self.LQRcost_dist,
                x0=self.Z0,
                constraints=constraints,
                tol=1e-3, # default 1e-3
                options={"disp": 5,
                         'maxiter': 400,
                         'linear_solver': 'ma57',
                         'hsllib': hsllib_path}
            )
        
        if version == "with_MHE":
            print(f"Executed MPC Solver Version: with_MHE")
            if(MHE_coeff is None): 
                print(f"User did not define MHE coefficient")
                return
            else:
                self.MHE_coeff = MHE_coeff
                constraints = [
                    {"type": "eq", "fun": self.extended_equality_constraints_fast},
                    {"type": "ineq", "fun": self.inequality_constraints_fast}
                ]
                self.sol = ipopt.minimize_ipopt(
                    fun=self.LQRcost_fast,
                    x0=self.Z0,
                    constraints=constraints,
                    tol=1e-3, # default 1e-3
                    options={"disp": 5,
                             'maxiter': 400,
                             'linear_solver': 'ma57',
                             'hsllib': hsllib_path}
                )

class MHE:
    def __init__(self, model: VesselModel, N, ns, nu, nc, dt, Qvec, Rvec, Qfvec, Pvec, c0, sref, uref):
        self.model = model
        self.N = N # this is the MPC window size, not the N of the entire trajectory
        self.ns = ns
        self.nu = nu
        self.nc = nc # number of hydrodynamic coefficients (default 13) + however many tweak variables
        self.dt = dt

        self.Qvec = Qvec
        
        self.Rvec = Rvec
        self.Qfvec = Qfvec
        self.Pvec = Pvec
        self.Q = np.diag(Qvec)
        self.R = np.diag(Rvec)
        self.Qf = np.diag(Qfvec)
        self.P = np.diag(Pvec) # weight matrix for coeff cost (to reduce difference between coefficients each run)

        self.c0 = c0 # initial guess (or warm start) of hydrodynamic coefficients
        self.sref = sref # reference trajectory MPC is following
        self.uref = uref # reference controls for estimated coefficient feed forward

        # coeff clamps
        self.c_min = -100
        self.c_max = 100

        self.Z0 = self.make_initial_guess()
        #assert len(self.Z0) == (N-1)*nc, "Length of initial guess does not match length calculated from given N and nc"

    def make_initial_guess(self):
        #coeff = np.zeros((self.N-1,self.nc))
        #coeff[0] = self.c0
        coeff = self.c0

        coeff_flat = coeff.flatten()
        guess = coeff_flat
        return guess
    
    def flat2vec(self, Z):
        coeff = Z.reshape((self.N-1, self.nc))
        return coeff
    
    def LQRcost_fast(self, Z):
        #coeff = self.flat2vec(Z)
        coeff = Z

        Sfwd = np.zeros((self.N,self.ns))
        Sfwd[0] = self.sref[0]
        for i in range(self.N-1):
            sn = Sfwd[i]
            un = self.uref[i]
            #cn = coeff[i]
            cn = coeff
            Sfwd[i+1] = self.model.extended_rk4(sn, un, cn, self.dt)

        S = Sfwd[:-1]

        cost = 0
        cost += np.sum(0.5 * np.vecdot((S-self.sref[:-1]), np.multiply(self.Qvec, S-self.sref[:-1])))
        cost += np.sum(0.5 * np.vecdot((Sfwd[-1]-self.sref[-1]), np.multiply(self.Qfvec, Sfwd[-1]-self.sref[-1])))
        cost += np.sum(0.5 * np.vecdot((coeff-self.c0), np.multiply(self.Pvec, coeff-self.c0)))
        #print(f"cost from states: {cost}")
        #cost_coeff = np.sum(0.5 * np.vecdot((coeff[1:]-coeff[:-1]), np.multiply(self.Pvec, coeff[1:]-coeff[:-1])))
        #cost_coeff = np.sqrt(np.sum(np.square(coeff)))
        #cost += cost_coeff
        #print(f"cost from coefficients: {cost_coeff}")

        return cost
    
    def inequality_constraints_fast(self, Z):
        coeff = Z

        constraints_min_controls = coeff - self.c_min
        constraints_max_controls = -coeff + self.c_max
        constraints_mass_values = coeff[13:15]
        
        constraints = np.concatenate((constraints_min_controls.flatten(), constraints_max_controls.flatten(), constraints_mass_values.flatten()))

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def exec_MPC(self):
        hsllib_path = f"{os.environ['CONDA_PREFIX']}/lib/x86_64-linux-gnu/libcoinhsl.so" # for local machine
        #hsllib_path = f"{os.environ['CONDA_PREFIX']}/lib64/libcoinhsl.so" # for NCSA

        constraints = [
            {"type": "ineq", "fun": self.inequality_constraints_fast}
        ]

        self.sol = ipopt.minimize_ipopt(
            fun=self.LQRcost_fast,
            x0=self.Z0,
            constraints=constraints,
            tol=1e-4,
            options={"disp": 5,
                     'maxiter': 400,
                     'linear_solver': 'ma57',
                     'hsllib': hsllib_path}
        )