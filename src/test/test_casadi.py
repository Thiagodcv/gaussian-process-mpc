from unittest import TestCase
import numpy as np
from casadi import *


class TestCasadi(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_casadi(self):
        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        nlp = {'x': vertcat(x, y, z), 'f': x ** 2 + 100 * z ** 2, 'g': z + (1 - x) ** 2 - y}
        S = nlpsol('S', 'ipopt', nlp)
        print(S)

        r = S(x0=[2.5, 3.0, 0.75], lbg=0, ubg=0)
        x_opt = r['x']
        print('x_opt: ', x_opt)

    def test_casadi_function_call(self):

        def f(a, b, c):
            return a ** 2 + 100 * c ** 2

        def g(a, b, c):
            return c + (1-a)**2 - b

        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        nlp = {'x': vertcat(x, y, z), 'f': f(x, y, z), 'g': g(x, y, z)}
        S = nlpsol('S', 'ipopt', nlp)
        print(S)

        r = S(x0=[2.5, 3.0, 0.75], lbg=0, ubg=0)
        x_opt = r['x']
        print('x_opt: ', x_opt)

    def test_casadi_numpy_call(self):
        H = np.identity(2)
        x = MX.sym('x', 2)

        def f(a):
            return a.T @ H @ a

        def g(a):
            return x[0] - 1

        nlp = {'x': x, 'f': f(x), 'g': g(x)}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0=[-1, -1], lbg=0, ubg=0)
        x_opt = r['x']
        print('x_opt: ', x_opt)
