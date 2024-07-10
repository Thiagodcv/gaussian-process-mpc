from src.models.pendulum import nom_model_om, nom_model_th, true_model_om, true_model_th, m, l, b, g, delta_t
from unittest import TestCase
import numpy as np
import torch


class TestPendulum(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_inputs(self):
        """
        Test inputting batches of observations into the method defining pendulum dynamics.
        """
        x = torch.tensor([1., 2.])
        u = torch.tensor([3.])

        x_batch = torch.tensor([[0., 0.5],
                                [1., 1.]])
        u_batch = torch.tensor([[0.5],
                                [0.7]])

        self.assertTrue(torch.linalg.norm(nom_model_th(x, u) - (2*delta_t + 1)).item() < 1e-5)
        self.assertTrue(torch.linalg.norm(nom_model_th(x_batch, u_batch) -
                                          torch.tensor([0.5 * delta_t, 1 + delta_t])).item() < 1e-5)

        self.assertTrue(torch.linalg.norm(nom_model_om(x, u) -
                                          (-g/l*np.sin(1)*delta_t + 2 + 1/(m*l**2)*3*delta_t)).item() < 1e-5)
        nom_model_om_batch_ans = torch.tensor([-g/l*np.sin(0)*delta_t + 0.5 + 1/(m*l**2)*0.5*delta_t,
                                               -g/l*np.sin(1)*delta_t + 1 + 1/(m*l**2)*0.7*delta_t])
        self.assertTrue(torch.linalg.norm(nom_model_om(x_batch, u_batch) - nom_model_om_batch_ans).item() < 1e-5)

        self.assertTrue(torch.linalg.norm(true_model_th(x, u) - (2*delta_t + 1)).item() < 1e-5)
        self.assertTrue(torch.linalg.norm(true_model_th(x_batch, u_batch) -
                                          torch.tensor([0.5 * delta_t, 1 + delta_t])).item() < 1e-5)

        self.assertTrue(torch.linalg.norm(true_model_om(x, u) -
                                          (-b/m*2*delta_t - g/l*np.sin(1)*delta_t + 2 + 1/(m*l**2)*3*delta_t)).item() < 1e-5)
        true_model_om_batch_ans = torch.tensor([-b/m*0.5*delta_t - g/l*np.sin(0)*delta_t + 0.5 + 1/(m*l**2)*0.5*delta_t,
                                                -b/m*1*delta_t - g/l*np.sin(1)*delta_t + 1 + 1/(m*l**2)*0.7*delta_t])
        self.assertTrue(torch.linalg.norm(true_model_om(x_batch, u_batch) - true_model_om_batch_ans).item() < 1e-5)
