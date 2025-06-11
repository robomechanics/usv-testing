from vessel_models import VesselModel
import numpy as np
import cyipopt as ipopt

class TrajOpt:
    def __init__(self, model: VesselModel, N, ns, nu, dt, Qvec, Rvec, Qfvec, s0, sf):
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
    
    def LQRcost(self, Z):
        states, inputs = self.flat2vec(Z)
        
        cost = 0
        for i in range(self.N-1):
            sn = states[i]
            un = inputs[i]
            # LQR Cost
            cost += 0.5 * (sn-self.sf) @ self.Q @ (sn-self.sf)
            cost += 0.5 * un @ self.R @ un
        
        cost += (states[-1]-self.sf) @ self.Qf @ (states[-1]-self.sf) # LQR terminal cost
        
        return cost
    
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
    
    def equality_constraints(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints = []

        # Initial and terminal constraints
        initial_constraint = states[0] - self.s0
        terminal_constraint = states[-1] - self.sf
        constraints.extend(initial_constraint)
        constraints.extend(terminal_constraint)

        # Dynamics constraints
        for i in range(self.N-1):
            sn = states[i]
            snp1 = states[i+1]
            un = inputs[i]
            #dyn_snp1 = self.model.euler(sn, un, self.dt)
            #dynamics_constraint = snp1 - dyn_snp1
            dynamics_constraint = self.model.hermite_simpson(sn, snp1, un, self.dt)
            constraints.extend(dynamics_constraint)

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
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
    
    def inequality_constraints(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints = []

        # Control clamping
        for i in range(self.N-1):
            un = inputs[i]
            constraints.extend(un-self.u_min)
            constraints.extend(self.u_max-un)

        # Simulated obstacles
        for i in range(self.N):
            sn = states[i]
            X = sn[0]
            Y = sn[1]
            oc1 = (X-0.25)**2 + (Y-0.25)**2 - 0.1**2
            oc2 = (X-0.6)**2 + (Y-0.4)**2 - 0.1**2
            oc3 = (X-0.75)**2 + (Y-0.75)**2 - 0.1**2
            constraints.extend([oc1, oc2, oc3])

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def inequality_constraints_fast(self, Z):
        states, inputs = self.flat2vec(Z)

        constraints_min_controls = inputs - self.u_min
        constraints_max_controls = -inputs + self.u_max
        X = states[:,0]
        Y = states[:,1]
        oc1 = (X-0.25)**2 + (Y-0.25)**2 - 0.1**2
        oc2 = (X-0.8)**2 + (Y-0.3)**2 - 0.25**2
        oc3 = (X-0.75)**2 + (Y-0.75)**2 - 0.1**2
        oc4 = (X-0.3)**2 + (Y-0.8)**2 - 0.25**2
        constraints = np.concatenate((constraints_min_controls.flatten(), constraints_max_controls.flatten(), oc1.flatten(), oc2.flatten(), oc3.flatten(), oc4.flatten()))

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
            tol=1e-3,
            options={"print_level": 5}
        )

class MPC:
    def __init__(self, model: VesselModel, N, ns, nu, dt, Qvec, Rvec, Qfvec, s0, sf, sref):
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

        assert np.all(np.isfinite(constraints)), "NaN or Inf in constraints"
        return np.array(constraints)
    
    def exec_MPC(self):
        constraints = [
            {"type": "eq", "fun": self.equality_constraints_fast},
            {"type": "ineq", "fun": self.inequality_constraints_fast}
        ]

        self.sol = ipopt.minimize_ipopt(
            fun=self.LQRcost_fast,
            x0=self.Z0,
            constraints=constraints,
            tol=1e-3,
            options={"disp": 5}
        )