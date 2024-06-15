from unittest import TestCase
import time
import numpy as np


class TestGaussianProcessRegression(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_matrix_inverse(self):
        # Results: 1000: 0.077s, 2000: 0.34s, 3000: 0.88s
        avg_time = []
        n_trials = 20
        for i in range(1, 3 + 1):
            print(i)
            size = (i * 1000, i * 1000)
            avg_time.append(0)
            for j in range(n_trials):
                start = time.time()
                np.linalg.inv(np.random.normal(size=size))
                end = time.time()
                avg_time[i - 1] += end - start
            avg_time[i - 1] = avg_time[i - 1] / n_trials
        print(avg_time)

    def test_nn_time(self):
        start = time.time()
        for i in range(100_000):
            a = 5 * 5
            a = 5 * 5
            a = 5 * 5
            if i == 5000:
                print(i)
        end = time.time()
        print(end-start)
