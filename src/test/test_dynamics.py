from src.dynamics import Dynamics
from src.models.pendulum import nom_model_th, nom_model_om, true_model_th, true_model_om, state_dim, action_dim
from src.mpc import RiskSensitiveMPC
from unittest import TestCase
import numpy as np
import torch
import cProfile
import time
import os


class TestDynamics(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_append_train_data(self):
        """
        Test to see if correct information is stored in each Gaussian process model. This test handles the case
        of a single observation.
        """
        dynamics = Dynamics(state_dim=2, action_dim=1, nominal_models=[nom_model_th, nom_model_om])
        self.assertTrue(len(dynamics.gpr_err) == 2)

        state = np.array([1., 0.5])
        action = np.array([0.8])

        next_state_th = true_model_th(state, action)
        next_state_om = true_model_om(state, action)
        next_state = np.array([next_state_th, next_state_om])

        dynamics.append_train_data(state, action, next_state)

        self.assertTrue(np.linalg.norm(dynamics.gpr_err[0].X_train.cpu().detach().numpy() -
                                       np.array([[1., 0.5, 0.8]])) < 1e-5)

        self.assertTrue(np.linalg.norm(dynamics.gpr_err[1].X_train.cpu().detach().numpy() -
                                       np.array([[1., 0.5, 0.8]])) < 1e-5)

        self.assertTrue(np.linalg.norm(dynamics.gpr_err[0].y_train.item() - next_state_th) < 1e-5)
        self.assertTrue(np.linalg.norm(dynamics.gpr_err[1].y_train.item() - next_state_om) < 1e-5)

    def test_append_train_data_batch(self):
        """
        Test to see if correct information is stored in each Gaussian process model. This test handles the case
        of more than one observation.
        """
        dynamics = Dynamics(state_dim=2, action_dim=1, nominal_models=[nom_model_th, nom_model_om])
        self.assertTrue(len(dynamics.gpr_err) == 2)

        state = np.array([[1., 0.5],
                          [1.5, 1.0]])  # Two observations (each row is an observation)
        action = np.array([[0.8],  # Can also do np.array([0.8, 1.2])
                           [1.2]])
        next_state_th = true_model_th(state, action)[:, None]
        next_state_om = true_model_om(state, action)[:, None]
        next_state = np.concatenate((next_state_th, next_state_om), axis=1)

        dynamics.append_train_data(state, action, next_state)

        self.assertTrue(np.linalg.norm(dynamics.gpr_err[0].X_train.cpu().detach().numpy() -
                                       np.array([[1., 0.5, 0.8],
                                                 [1.5, 1.0, 1.2]])) < 1e-5)

        self.assertTrue(np.linalg.norm(dynamics.gpr_err[1].X_train.cpu().detach().numpy() -
                                       np.array([[1., 0.5, 0.8],
                                                 [1.5, 1.0, 1.2]])) < 1e-5)

        self.assertTrue(np.linalg.norm(dynamics.gpr_err[0].y_train.cpu().detach().numpy() - next_state_th) < 1e-5)
        self.assertTrue(np.linalg.norm(dynamics.gpr_err[1].y_train.cpu().detach().numpy() - next_state_om) < 1e-5)

    def test_forward_propagate_nominal_model(self):
        """
        TODO: Nominal model functionality not fully implemented yet. Will return to this afterwards.
        Test the forward propagate algorithm using a nominal model.
        """
        dynamics = Dynamics(state_dim=2, action_dim=1, nominal_models=[nom_model_th, nom_model_om])
        self.assertTrue(len(dynamics.gpr_err) == 2)

        state = np.array([[1., 0.5],
                          [1.5, 1.0]])  # Two observations (each row is an observation)
        action = np.array([[0.8],
                           [1.2]])
        next_state_th = true_model_th(state, action)[:, None]  # Some data points just so we can get K matrix
        next_state_om = true_model_om(state, action)[:, None]
        next_state = np.concatenate((next_state_th, next_state_om), axis=1)
        dynamics.append_train_data(state, action, next_state)

        horizon = 5
        init_state = np.array([0., 0])
        state_means, state_covars = dynamics.forward_propagate(horizon=horizon,
                                                               curr_state=init_state,
                                                               actions=np.zeros((horizon, action_dim)))
        print(state_means)

    def test_forward_propagate(self):
        """
        Test forward propagate algorithm when not using a nominal model.
        """
        num_train = 10
        dynamics = Dynamics(state_dim=2, action_dim=1, nominal_models=None)
        dynamics.gpr_err[0].set_sigma_n(0.1)
        dynamics.gpr_err[1].set_sigma_n(0.1)

        # Generate states encountered in training data
        state_th = np.random.uniform(low=0, high=np.pi, size=(num_train, 1))
        state_om = np.random.uniform(low=-2, high=2, size=(num_train, 1))
        state = np.concatenate((state_th, state_om), axis=1)

        # Generate actions from training data
        action = np.random.uniform(low=-2, high=2, size=(num_train, 1))

        # Generate next actions from training data
        next_state_th = true_model_th(state, action)[:, None]
        next_state_om = true_model_om(state, action)[:, None]
        next_state = np.concatenate((next_state_th, next_state_om), axis=1)

        dynamics.append_train_data(state, action, next_state)

        horizon = 1
        init_state = np.array([0., 0.5])
        start = time.time()
        state_means, state_covars = dynamics.forward_propagate(horizon=horizon,
                                                               curr_state=init_state,
                                                               actions=np.zeros((horizon, action_dim)))
        end = time.time()
        print("time: ", end-start)
        # run_str = 'dynamics.forward_propagate(horizon=horizon, curr_state=init_state, actions=np.zeros((horizon, action_dim)))'
        # cProfile.runctx(run_str, globals(), locals())

    def test_forward_propagate_torch(self):
        """
        Test the Torch implementation of the forward propagate algorithm when not using a nominal model.
        Make sure it agrees with NumPy implementation.

        num_train = 10 and horizon = 1: 0.028s
        num_train = 1000 and horizon = 1: 0.096s
        num_train = 1000 and horizon = 10: 0.82s
        num_train = 5000 and horizon = 10: 12.5s
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # To show real time spent on .time()

        # Get GPR model set up
        num_train = 10
        dynamics = Dynamics(state_dim=2, action_dim=1, nominal_models=None)
        dynamics.gpr_err[0].set_sigma_n(0.1)
        dynamics.gpr_err[1].set_sigma_n(0.1)

        # Generate states encountered in training data
        state_th = np.random.uniform(low=0, high=np.pi, size=(num_train, 1))
        state_om = np.random.uniform(low=-2, high=2, size=(num_train, 1))
        state = np.concatenate((state_th, state_om), axis=1)

        # Generate actions from training data
        action = np.random.uniform(low=-2, high=2, size=(num_train, 1))

        # Generate next actions from training data
        next_state_th = true_model_th(state, action)[:, None]
        next_state_om = true_model_om(state, action)[:, None]
        next_state = np.concatenate((next_state_th, next_state_om), axis=1)

        dynamics.append_train_data(state, action, next_state)

        # Run forward propagation method
        horizon = 2
        init_state = np.array([0., 0.5])

        # Run NumPy version (only do for small num_train)
        state_means_np, state_covars_np = dynamics.forward_propagate(horizon=horizon,
                                                                     curr_state=init_state,
                                                                     actions=np.zeros((horizon, action_dim)))

        init_state = torch.tensor(init_state, device=device)

        start = time.time()
        state_means, state_covars = dynamics.forward_propagate_torch(horizon=horizon,
                                                                     curr_state=init_state,
                                                                     actions=torch.zeros((horizon, action_dim),
                                                                                         device=device))
        end = time.time()
        print("time: ", end-start)

        # run_str = 'dynamics.forward_propagate_torch(horizon=horizon, curr_state=init_state, actions=torch.zeros((horizon, action_dim), device=device))'
        # cProfile.runctx(run_str, globals(), locals())

        self.assertTrue(np.linalg.norm(state_means[0].cpu().detach().numpy() - state_means_np[0, :]) < 1e-7)
        self.assertTrue(np.linalg.norm(state_means[1].cpu().detach().numpy() - state_means_np[1, :]) < 1e-7)
        self.assertTrue(np.linalg.norm(state_means[2].cpu().detach().numpy() - state_means_np[2, :]) < 1e-7)

        self.assertTrue(np.linalg.norm(state_covars[0].cpu().detach().numpy() - state_covars_np[0, :, :]) < 1e-7)
        self.assertTrue(np.linalg.norm(state_covars[1].cpu().detach().numpy() - state_covars_np[1, :, :]) < 1e-7)
        self.assertTrue(np.linalg.norm(state_covars[2].cpu().detach().numpy() - state_covars_np[2, :, :]) < 1e-7)

    def test_forward_propagate_torch_mc(self):
        """
        Ensure Dynamics.forward_propagate_torch appears to match results from MC simulation. Note
        that the results shouldn't necessarily be arbitrarily close for large N, since mean/variance/covariance_prop
        methods are really just approximations when used recursively.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_train = 30
        s_min = -10
        s_max = 10
        a_min = -1
        a_max = 1

        def f(s, a):
            return s + a

        state = np.random.uniform(s_min, s_max, num_train)[:, None]
        action = np.random.uniform(a_min, a_max, num_train)[:, None]

        next_state = f(state, action)

        Q = 2 * np.identity(1)
        R = np.array([[0]])
        R_delta = np.array([[0]])
        gamma = 1e-5  # Negative gamma is risk-averse, positive gamma is risk-seeking
        horizon = 5
        state_dim = 1
        action_dim = 1
        mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

        mpc.dynamics.gpr_err[0].set_sigma_n(1e-5)  # Recall method doesn't automatically make Ky get rebuilt
        mpc.dynamics.gpr_err[0].set_sigma_f(5.)
        mpc.dynamics.gpr_err[0].set_lambdas([2., 2.])
        mpc.dynamics.append_train_data(state, action, next_state)
        mpc.set_ub([a_max])
        mpc.set_lb([a_min])
        mpc.set_xref(np.array([0.]))
        mpc.set_uref(np.array([0.]))

        curr_state = torch.tensor([5.], device=device).type(torch.float64)
        actions = torch.tensor([-1, -1, -1, -1, -1], device=device)[:, None].type(torch.float64)

        state_means, state_covars = mpc.dynamics.forward_propagate_torch(horizon=horizon,
                                                                         curr_state=curr_state, actions=actions)

        print("forward_prop means: ", torch.concatenate(state_means, axis=0).cpu().detach().numpy())
        print("forward_prop vars: ", torch.concatenate(state_covars, axis=0).cpu().detach().numpy().flatten())

        # MC Method
        num_iters = 2000
        x0_mean = curr_state.cpu().detach().numpy()
        x0 = np.random.normal(loc=x0_mean, scale=np.sqrt(1e-3), size=num_iters)[None, :]
        x_mat = np.zeros((horizon, num_iters))
        x_mat = np.concatenate((x0, x_mat), axis=0)

        actions_mean = actions.cpu().detach().numpy()  # Actually, actions also have a variance of 1e-3
        actions = np.random.multivariate_normal(mean=actions_mean.flatten(),
                                                cov=1e-3*np.identity(horizon),
                                                size=num_iters).T

        for i in range(horizon):
            for j in range(num_iters):
                z = np.concatenate((x_mat[i, j][None], actions[i, j][None]))
                x_mean, x_var = mpc.dynamics.gpr_err[0].predict_latent_vars(z[None, :], covar=True, targets=True)
                x_mat[i+1, j] = np.random.normal(loc=x_mean[0, 0], scale=np.sqrt(x_var[0, 0]))

        x_means = np.mean(x_mat, axis=1)
        x_vars = np.var(x_mat, axis=1)
        print("MC means: ", x_means)
        print("MC vars: ", x_vars)
        # Results seem to be pretty close!
