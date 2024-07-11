from src.dynamics import Dynamics
from src.models.pendulum import nom_model_th, nom_model_om, true_model_th, true_model_om, state_dim, action_dim
from unittest import TestCase
import numpy as np
import torch


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

    def test_forward_propagate(self):
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
