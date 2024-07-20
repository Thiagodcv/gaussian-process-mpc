from unittest import TestCase
import cyipopt
import numpy as np
import torch


class TestIPOPT(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ipopt(self):
        """
        Total seconds in just IPOPT: 0.142s.
        Total seconds in IPOPT + Torch: 2.627s.

        Quite slow... Perhaps using JAX would be faster?
        """

        x0 = [1.0, 5.0, 5.0, 5.0]

        lb = [1.0, 1.0, 1.0, 1.0]
        ub = [5.0, 5.0, 5.0, 5.0]

        cl = [25.0, 40.0]
        cu = [2.0e19, 40.0]

        nlp = cyipopt.Problem(
            n=len(x0),
            m=len(cl),
            problem_obj=hs071(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )

        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-7)

        x, info = nlp.solve(x0)
        print(x)  # Optimal solution should be (1.0, 4.743, 3.821, 1.379)


class hs071(object):

    def __init__(self):
        self.curr_obj = None
        self.curr_grad = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def objective(self, x):
        """
        The callback for calculating the objective.

        Parameters:
        ----------
        x: np.array

        Return:
        ------
        scalar
        """
        x = torch.tensor(x, device=self.device).requires_grad_(True)
        obj = x[0] * x[3] * torch.sum(x[0:3]) + x[2]
        obj.backward()

        self.curr_obj = obj
        self.curr_grad = x.grad.cpu().detach().numpy()

        return obj.item()

    def gradient(self, x):
        """
        The callback for calculating the gradient of the objective function.

        Parameters:
        ----------
        x: np.array

        Return:
        ------
        np.array
        """
        if self.curr_obj is None:
            self.objective(x)

        return self.curr_grad

    def constraints(self, x):
        #
        # The callback for calculating the constraint functions
        #
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian of constraint functions
        #
        return np.concatenate((np.prod(x) / x, 2*x))

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
