from unittest import TestCase
import numpy as np
from casadi import *


class TestCasadi(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Casadi(self):
        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        nlp = {'x': vertcat(x, y, z), 'f': x ** 2 + 100 * z ** 2, 'g': z + (1 - x) ** 2 - y}
        S = nlpsol('S', 'ipopt', nlp)
        print(S)

        r = S(x0=[2.5, 3.0, 0.75], lbg=0, ubg=0)
        x_opt = r['x']
        print('x_opt: ', x_opt)

