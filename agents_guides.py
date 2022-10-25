import json
import math
import random
from collections import defaultdict
from statistics import mean

from agents import GuideAgent
from feature_extractor import FeatureExtractor


class GuideABT(GuideAgent):
    alpha = 0.5
    beta = 0.1
    theta = 3

    def __init__(self, uid, pos, model, mode, positions_set):
        super().__init__(uid, pos, model, positions_set)
        self.mode = mode

        self.direction_change_timer = 0

        if mode == "A":
            self.get_best_position = self.get_best_position_a
        elif mode == "B":
            self.get_best_position = self.get_best_position_b
        elif mode == "none":
            pass

    def step(self):
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

    def get_best_position_a(self, ):
        guides_positions = self.model.guides_positions.copy()
        guides_positions.remove(self.pos)

        legal_positions = self.get_legal_positions()

        if guides_positions == set():
            return

        legal_positions_with_goal = {}
        for pos in legal_positions:
            for s in self.model.sensors:
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
        guides_positions = self.model.guides_positions.copy()
        guides_positions.remove(self.pos)

        return 0, 0


class GuideQLearning(GuideAgent):
    def __init__(self, uid, pos, model, positions_set, epsilon=0.25, gamma=0.8, alpha=0.2, num_training=0,
                 extractor=None, weights=None):

        super().__init__(uid, pos, model, positions_set)
        self.episodes_so_far = 0

        self.epsilon = epsilon
        self.gamma = gamma  # aka discount factor
        self.alpha = alpha
        self.num_training = num_training

        # self.index = 0  # This is always Pacman
        self.epsilon_min = 0.20
        self.epsilon_diff = (epsilon - self.epsilon_min) / (num_training + 0.00000000000000001)

        if extractor is None:
            self.extractor = FeatureExtractor(self.model, self)

        self.weights = weights
        # if weights is None:
        #     self.weights = defaultdict(lambda: 0.0)

    def step(self):
        # Move section
        legal_actions = list(self.get_legal_actions().keys())

        if not legal_actions:
            return

        best_action = self.compute_action_from_q_values(legal_actions)
        legal_actions.remove(best_action)

        actions = [best_action] + legal_actions
        actions_weights = [1 - self.epsilon] + ([self.epsilon] * len(legal_actions))

        action_to_take = random.choices(actions, actions_weights)[0]
        position_to_go = self.extractor.action_to_position(self.pos, action_to_take)

        current_state = self.extractor.get_state()
        next_state = self.extractor.get_next_state(action_to_take)
        reward = self.extractor.get_reward(current_state, next_state)

        self.update(action_to_take, reward)
        self.move(position_to_go)

        # Broadcast section
        closest_exit_id, _ = self.get_closest_exit()
        self.broadcast_exit_id(closest_exit_id)

    def compute_action_from_q_values(self, legal_actions):
        actions = []
        for action in legal_actions:
            actions.append((action, self.get_q_value(action)))

        best_action = max(actions, key=lambda x: x[1])
        return best_action[0]

    def get_q_value(self, action):
        feats = self.extractor.get_next_state(action)
        q_val = 0
        for k in feats.keys():
            q_val += feats[k] * self.weights[k]
        return q_val

    def update(self, action, reward):
        max_Q_sa_prim = self.compute_value_from_a_values()
        Q_sa = self.get_q_value(action)
        diff = reward + (self.gamma * max_Q_sa_prim) - Q_sa

        feats = self.extractor.get_next_state(action)
        for k in feats.keys():
            self.weights[k] = self.weights[k] + (self.alpha * diff * feats[k])

    def compute_value_from_a_values(self):
        legal_actions = self.get_legal_actions()
        if not legal_actions:
            return 0.0

        values = []
        for action in legal_actions:
            values.append(self.get_q_value(action))

        return max(values)

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self):
        return self.compute_value_from_a_values()

    def get_weights(self):
        return self.weights

    def final(self, reward):
        # final reward update

        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_diff

        if self.episodes_so_far == self.num_training:
            with open("output/weights.txt", "w") as f:
                f.write(json.dumps(self.get_weights()))
