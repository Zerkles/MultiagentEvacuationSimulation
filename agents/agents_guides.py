import json
import math
import random
from collections import defaultdict
from statistics import mean
from typing import Tuple, Dict

from agents.agents import GuideAgent, Evacuee
from agents.feature_extractor import FeatureExtractor


class GuideABT(GuideAgent):
    alpha = 0.5
    beta = 0.1
    theta = 3

    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: random.Random, mode: str, args: Dict = None):
        super().__init__(uid, pos, random_seed, args)
        self.mode = mode

        self.direction_change_timer = 0

        if mode == "A":
            self.get_best_position = self.get_best_position_a
        elif mode == "B":
            self.get_best_position = self.get_best_position_b
        elif mode == "None":
            self.step = super().step

    def step(self, state):
        # Escape broadcast
        closest_exit_id, _ = self.get_closest_exit()
        self.broadcast_exit_id(closest_exit_id)

        # Theta timer
        if self.direction_change_timer > 0:
            self.direction_change_timer -= 1
            return

        # Moving
        best_position = self.get_best_position()

        self.move(best_position)

    def get_best_position_a(self):
        guides_positions = self.model.guides_positions.copy()
        guides_positions.remove(self.pos)

        legal_positions = self.get_legal_positions()

        if guides_positions == set():
            return

        legal_positions_with_goal = {}
        for pos in legal_positions:
            for s in Sensor.positions:
                direction_pull = s.evacuees_in_area / (math.dist(pos, s.pos) + 0.0001)  # to avoid zero division

                direction_repulsion = 0
                if guides_positions != set():
                    direction_repulsion = GuideAgent.beta / mean(
                        [math.dist(g_pos, pos) for g_pos in guides_positions])

                goal = GuideAgent.alpha * direction_pull + (1 - GuideAgent.alpha) * direction_repulsion

            legal_positions_with_goal.update({pos: goal})

        best_position = min(legal_positions_with_goal, key=legal_positions_with_goal.get)

        return best_position

    def get_best_position_b(self):
        guides_positions = GuideAgent.positions.copy()
        guides_positions.remove(self.pos)

        return 0, 0


class GuideQLearning(GuideAgent):
    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: random.Random, epsilon: float = 0.0,
                 gamma: float = 0.8, alpha: float = 0.2, extractor=None, weights=None):

        super().__init__(uid, pos, random_seed)
        self.score = 100

        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.weights = weights
        self.last_action = None

        if extractor is None:
            self.extractor = FeatureExtractor(self.unique_id)

        if weights is None:
            self.weights = defaultdict(lambda: 0.0)

    def step(self, state) -> str:
        # Move section
        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        ghost_agents = state.ghost_agents
        legal_actions = state.grid.get_legal_actions(pos, ghost_agents)

        best_action = self.compute_action_from_q_values(state, legal_actions)
        legal_actions.remove(best_action)

        actions = [best_action] + legal_actions
        actions_weights = [1 - self.epsilon] + ([self.epsilon] * len(legal_actions))

        action_to_take = random.choices(actions, actions_weights)[0]

        self.last_action = action_to_take

        return action_to_take

    def compute_action_from_q_values(self, state, legal_actions):
        actions = []
        for action in legal_actions:
            actions.append((action, self.get_q_value(state, action)))

        best_action = max(actions, key=lambda x: x[1])
        return best_action[0]

    def get_q_value(self, state, action):
        feats = self.extractor.get_features(state, action)
        q_val = 0

        for k in feats.keys():
            q_val += feats[k] * self.weights[k]
        return q_val

    def update(self, state, action, next_state, reward):
        max_Q_sa_prim = self.compute_value_from_q_values(next_state)
        Q_sa = self.get_q_value(state, action)
        diff = reward + (self.gamma * max_Q_sa_prim) - Q_sa

        feats = self.extractor.get_features(state, action)
        for k in feats.keys():
            self.weights[k] = self.weights[k] + (self.alpha * diff * feats[k])

        self.extractor.update_extractor(state, action)
        self.score += reward

        # print(self.score)
        # print("state: ", state.schedule.agents_by_breed[type(self)][self.unique_id].pos, action)
        # print("next: ", next_state.schedule.agents_by_breed[type(self)][self.unique_id].pos)
        # print(reward)

    def compute_value_from_q_values(self, state):
        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        ghost_agents = state.ghost_agents
        legal_actions = state.grid.get_legal_actions(pos, ghost_agents)

        if not legal_actions:
            return 0.0

        values = []
        for action in legal_actions:
            values.append(self.get_q_value(state, action))

        return max(values)

    def get_exit(self):
        return 1

    def get_policy(self, state):
        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        ghost_agents = state.ghost_agents
        legal_actions = state.grid.get_legal_actions(pos, ghost_agents)

        return self.compute_action_from_q_values(state, legal_actions)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)

    def get_weights(self):
        return self.weights

    def on_remove(self, state):
        if state.schedule.agents_by_breed[Evacuee] == 0:
            last_reward = 300
        else:
            last_reward = -100

        self.update(state, self.last_action, state, last_reward)
        return {'epsilon': self.epsilon, 'alpha': self.alpha, 'gamma': self.gamma, 'weights': self.weights}
