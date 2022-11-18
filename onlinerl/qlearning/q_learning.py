# -*- coding: utf-8 -*-
import random
import numpy as np
import pickle
from structlog import get_logger

from onlinerl.base import Agent
from onlinerl import logger

logger = get_logger(__name__)


class QAgent(Agent):
    def __init__(
        self,
        actions,
        alpha=0.4,
        gamma=0.9,
        random_seed=0,
        model_db=None,
        score_db=None,
        his_db=None,
    ):
        """
        The Q-values will be stored in a dictionary. Each key will be of the format: ((x, y), a).
        params:
            actions (list): A list of all the possible action values.
            alpha (float): step size
            gamma (float): discount factor
        """
        self._model_db = model_db
        self._score_db = score_db
        self._his_db = his_db
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        random.seed(random_seed)
        logger.info(
            f"QAgent init actions: {self.actions}", alpha=self.alpha, gamma=self.gamma
        )

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
        self, state, action, reward, next_state, model_id=None, reward_type="avg"
    ):
        """
        Q-Learning update
        """
        model_key = f"{model_id}:qvalue"
        score_key = f"{model_id}:{state}:Qscore"

        _Q_dict = self._model_db.get(model_key)
        if _Q_dict is None:
            Q_dict = {}
        else:
            Q_dict = pickle.loads(_Q_dict)

        if reward_type == "avg":
            reward = self.reward_action(
                state, action, reward=reward, model_id=model_id, init_model="no"
            )
        else:
            reward = float(reward)
        q_next = self.get_Q_value(
            Q_dict, state=next_state, action=self.greedy_action_selection(next_state)[0]
        )

        q_current = Q_dict.get(
            (state, action), None
        )  # If this is the first time the state action pair is encountered
        if q_current is None:
            _Q_score = reward
        else:
            _Q_score = q_current + (
                self.alpha * (reward + self.gamma * q_next - q_current)
            )

        Q_dict[(state, action)] = _Q_score
        self._model_db.set(model_key, pickle.dumps(Q_dict))
        Q_score = "-{}".format(_Q_score)
        self._score_db.zadd(score_key, Q_score, str(action))

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

    def getNextState(self, cur_state, action):
        pass

    def reward_action(self, state, action, reward=None, model_id=None, init_model="no"):

        success_key = f"{model_id}:{state}:{action}:successes"
        key_tries = f"{model_id}:{state}:{action}:tries"

        if reward is None:
            reward = 1.0
        reward_his = self._his_db.get(success_key)
        if reward_his is None:
            _reward = 0.0
        else:
            _reward = float(reward_his)
        _reward += reward
        self._his_db.set(success_key, str(_reward))

        state_action_tries = self._his_db.get(key_tries)
        if state_action_tries is None:
            state_action_tries = 1
        else:
            state_action_tries = int(state_action_tries)

        if init_model == "yes":
            # 初始化模型时默认曝光
            self.add_action_tries(state, action, model_id)

        self._his_db.set(success_key, str(_reward))

        return None if state_action_tries == 0 else _reward / state_action_tries

    def add_action_tries(self, state: str, action: str, reward, model_id: str) -> None:
        key_tries = f"{model_id}:{state}:{action}:tries"
        action_tries = self._his_db.incr(key_tries)
        return action_tries
