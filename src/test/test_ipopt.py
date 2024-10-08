from unittest import TestCase
import cyipopt
import numpy as np


class TestIPOPT(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ipopt(self):
        x0 = [1.0, 5.0, 5.0, 5.0]  # ,1.0]

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
        pass

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        print('objective')
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        #
        # The callback for calculating the gradient of objective function
        #
        print('gradient')
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraint functions
        #
        print('constraints')
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian of constraint functions
        #
        print('jacobian')
        return np.concatenate((np.prod(x) / x, 2*x))

    # def hessianstructure(self):
    #     #
    #     # Callback function that accepts no parameters and returns the sparsity structure
    #     # of the Hessian of the lagrangian (the row and column indices only)
    #     #
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
    #     print('hessianstructure')
    #     global hs
    #
    #     # hs = sps.coo_matrix(np.tril(np.ones((4, 4))))
    #     hs = scipy.sparse.coo_matrix(np.tril(np.ones((4, 4))))
    #     return (hs.col, hs.row)
    #
    # def hessian(self, x, lagrange, obj_factor):
    #     #
    #     # The callback for calculating the Hessian of the Lagrangian
    #     #
    #     print('hessian')
    #     H = obj_factor*np.array((
    #             (2*x[3], 0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
    #
    #     H += lagrange[0]*np.array((
    #             (0, 0, 0, 0),
    #             (x[2]*x[3], 0, 0, 0),
    #             (x[1]*x[3], x[0]*x[3], 0, 0),
    #             (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
    #
    #     H += lagrange[1]*2*np.eye(4)
    #
    #     #
    #     # Note:
    #     #
    #     #
    #     return H[hs.row, hs.col]

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
