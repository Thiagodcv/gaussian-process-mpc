from src.models.pendulum import nom_model_om, nom_model_th, true_model_om, true_model_th
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
        u = torch.tensor([3., 4.])

        x_batch = torch.tensor([[0., 0.5],
                                [1., 1.]])
        u_batch = torch.tensor([[0.5],
                                [0.5]])

        print(nom_model_th(x, u))
        print(nom_model_th(x_batch, u_batch))

        print(nom_model_om(x, u))
        print(nom_model_om(x_batch, u_batch))

        print(true_model_th(x, u))
        print(true_model_th(x_batch, u_batch))

        print(true_model_om(x, u))
        print(true_model_om(x_batch, u_batch))
