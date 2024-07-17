from src.dynamics import Dynamics
from src.models.pendulum import nom_model_th, nom_model_om, true_model_th, true_model_om, state_dim, action_dim
from unittest import TestCase
import numpy as np
import torch
import cProfile
import time


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
        num_train = 10 and horizon = 1: 0.028s
        num_train = 1000 and horizon = 1: 0.096s
        num_train = 1000 and horizon = 10: 0.82s
        num_train = 5000 and horizon = 10: 12.5s
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Get GPR model set up
        num_train = 3000
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
        horizon = 10
        init_state = torch.tensor([0., 0.5], device=device)
        start = time.time()
        state_means, state_covars = dynamics.forward_propagate_torch(horizon=horizon,
                                                                     curr_state=init_state,
                                                                     actions=torch.zeros((horizon, action_dim),
                                                                                         device=device))
        end = time.time()
        print("time: ", end-start)
        # run_str = 'dynamics.forward_propagate_torch(horizon=horizon, curr_state=init_state, actions=torch.zeros((horizon, action_dim), device=device))'
        # cProfile.runctx(run_str, globals(), locals())
