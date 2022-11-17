import random

class Agent:

    def __init__(self, actions, alpha = 0.4, gamma=0.9, random_seed=0, model_db= None):
        """
        The Q-values/Sarsa/Expected Sarsa will be stored in a dictionary. Each key will be of the format: ((x, y), a). 
        params:
            actions (list): A list of all the possible action values.
            alpha (float): step size
            gamma (float): discount factor
        """
        self.Q = {}
        self._model_db = model_db
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        random.seed(random_seed)

    def get_Q_value(self, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        raise NotImplementedError("This method must be implemented.")
    def act(self, state, epsilon=0.1):
        raise NotImplementedError("This method must be implemented.")

    def learn(self, state, action, reward, next_state):
        """
        Q-Learning update
        """
        raise NotImplementedError("This method must be implemented.")
    def greedy_action_selection(self, state):
        """
        Selects action with the highest Q-value for the given state.
        """
        raise NotImplementedError("This method must be implemented.")