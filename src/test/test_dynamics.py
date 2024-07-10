from src.dynamics import Dynamics
from src.models.pendulum import nom_model_th, nom_model_om, true_model_th, true_model_om
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
