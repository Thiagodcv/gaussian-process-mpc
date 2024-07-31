from unittest import TestCase
import numpy as np
import torch


class TestPytorch(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_autograd_bug(self):
        """
        Try to recreate a bug I encountered having to do with the autograd graph.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.tensor([5.], requires_grad=True, device=device)

        # Next 6 lines causes this bug:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation.
        # y = torch.tensor([0., 0.], device=device)
        # y[0] = x
        # y[1] = y[0]**2
        #
        # y[1].backward()
        # print(x.grad)

        # Instead do
        # y = list()
        # y.append(x)
        # y.append(y[0]**2)
        # cost = y[0] + y[1]
        # cost.backward()
        # print(x.grad)

        # How about concatenation?
        y = x
        b = y**2
        y = torch.concatenate((y, b))
        y[1].backward()
        print(x.grad)
