from src.gpr import GaussianProcessRegression


class Dynamics(object):

    def __init__(self, state_dim, action_dim, nominal_model):
        """
        Parameters:
        ----------
        state_dim: int
            The dimension of the state space
        action_dim: int
            The dimension of the action space
        nominal_model: function
            The nominal model of the system
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nominal_model = nominal_model
        self.gpr_err = [GaussianProcessRegression(x_dim=self.state_dim + self.action_dim) for _ in range(state_dim)]
