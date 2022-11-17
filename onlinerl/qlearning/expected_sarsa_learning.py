# ExpectedSarsaAgent.py

import random
import numpy as np
import pickle
from structlog import get_logger

from onlinerl.base import Agent
from onlinerl import logger

logger = get_logger(__name__)


class ExpectedSarsaAgent(Agent):
    def __init__(
        self,
        actions,
        epsilon=0.1,
        alpha=0.4,
        gamma=0.9,
        num_state=None,
        num_actions=None,
        action_space=None,
        model_db=None,
        score_db=None,
    ):
        """
        Constructor
        Args:
                epsilon: The degree of exploration
                gamma: The discount factor
                num_state: The number of states
                num_actions: The number of actions
                action_space: To call the random action
        """
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space
        self._model_db = model_db
        self._score_db = score_db

    def get_Q_value(self, Q_dict, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return Q_dict.get(
            (state, action), 0.0
        )  # Return 0.0 if state-action pair does not exist

    def act(self, state, epsilon=0.1, model_id=None, topN=1):
        # Choose a random action
        explore = np.random.binomial(1, epsilon)
        if explore == 1:
            # action = random.choice(self.actions)
            action = self.get_random_action(topN)
        # Choose the greedy action
        else:
            action = self.greedy_action_selection(state, model_id, topN)
            if len(action) < 1:
                action = self.get_random_action(topN)

        return action

    def learn(
        self, prev_state, next_state, reward, prev_action, next_action, model_id=None
    ):
        """
        Update the action value function using the Expected SARSA update.
        Q(S, A) = Q(S, A) + alpha(reward + (pi * Q(S_, A_) - Q(S, A))
        Args:
                prev_state: The previous state
                next_state: The next state
                reward: The reward for taking the respective action
                prev_action: The previous action
                next_action: The next action
        Returns:
                None
        """
        model_key = f"{model_id}:qvalue"
        score_prev_state = f"{model_id}:{prev_state}:Qscore"
        _Q_dict = self._model_db.get(model_key)

        if _Q_dict is None:
            Q_dict = {}
        else:
            Q_dict = pickle.loads(_Q_dict)

        predict = self.get_Q_value(Q_dict, prev_state, prev_action)

        expected_q = 0
        q_max_action = self.greedy_action_selection(
            next_state, model_id=model_id, topN=1
        )[0]
        q_max = self.get_Q_value(Q_dict, next_state, q_max_action)

        greedy_actions = 0
        for i in range(self.num_actions):
            if self.get_Q_value(Q_dict, next_state, self.actions[i]) == q_max:
                greedy_actions += 1

        non_greedy_action_probability = self.epsilon / self.num_actions
        greedy_action_probability = (
            (1 - self.epsilon) / greedy_actions
        ) + non_greedy_action_probability

        for i in range(self.num_actions):
            next_state_ai = self.get_Q_value(Q_dict, next_state, self.actions[i])
            if next_state_ai == q_max:
                expected_q += next_state_ai * greedy_action_probability
            else:
                expected_q += next_state_ai * non_greedy_action_probability

        target = reward + self.gamma * expected_q
        Q_dict[(prev_state, prev_action)] += self.alpha * (target - predict)
        # self.Q[prev_state, prev_action] += self.alpha * (target - predict)

        _Q_score = Q_dict[(prev_state, prev_action)]
        self._model_db.set(model_key, pickle.dumps(Q_dict))
        Q_score = "-{}".format(_Q_score)
        self._score_db.zadd(score_prev_state, Q_score, str(prev_action))

    def greedy_action_selection(self, state, model_id=None, topN=1):
        """
        Selects action with the highest Q-value for the given state.
        """
        # Get all the Q-values for all possible actions for the state
        maxQ_action_list = self.get_maxQ(state, model_id, topN, withscores=False)
        if len(maxQ_action_list) < 1:
            maxQ_action_list = self.get_random_action(topN)
        return maxQ_action_list

    def get_maxQ(self, state, model_id, topN, withscores=False):
        score_key = f"{model_id}:{state}:Qscore"
        if withscores:
            score_list = self._score_db.zrange(
                score_key, "0", str(topN - 1), "withscores"
            )
        else:
            score_list = self._score_db.zrange(score_key, "0", str(topN - 1))
        return score_list

    def get_random_action(self, topN):
        if topN > len(self.actions):
            raise Exception("topN is longer than len of self.actions")
        action_list = np.random.choice(
            self.actions, size=topN, replace=False, p=None
        ).tolist()
        return action_list
