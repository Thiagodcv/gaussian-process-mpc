from src.dynamics import Dynamics
from src.models.pendulum import nom_model_om, nom_model_th, true_model_om, true_model_th
from unittest import TestCase
import numpy as np
import torch


class TestDynamics(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_append_train_data(self):
        dynamics = None
