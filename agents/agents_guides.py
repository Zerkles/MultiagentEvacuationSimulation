import random
from collections import defaultdict
from typing import Tuple, Dict, List

from agents.agents import GuideAgent, Exit
from agents.feature_extractor import FeatureExtractor
from simulation.simulation_state import SimulationState


class GuideQLearning(GuideAgent):
    def __init__(self, uid: int, pos: Tuple[int, int], random_seed: random.Random, epsilon: float = 0.0,
                 gamma: float = 0.8, alpha: float = 0.2, extractor: FeatureExtractor = None,
                 weights: Dict = None) -> None:

        super().__init__(uid, pos, random_seed)
        self.lifepoints = 100
        self.score = 0

        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.alpha = alpha  # learning rate
        self.weights = weights
        self.last_feats = None

        if extractor is None:
            self.extractor = FeatureExtractor(self.unique_id)

        if weights is None:
            self.weights = defaultdict(lambda: 0.0)

    def step(self, state: SimulationState) -> str:
        # Move section
        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        legal_actions = state.grid.get_legal_actions(pos, state.ghost_agents)

        if legal_actions == {}:
            return "MM"

        best_action = self.compute_action_from_q_values(state, legal_actions)
        legal_actions.remove(best_action)

        actions = [best_action] + legal_actions
        actions_weights = [1 - self.epsilon] + ([self.epsilon] * len(legal_actions))
        action_to_take = random.choices(actions, actions_weights)[0]

        self.last_feats = self.extractor.get_features(state, action_to_take)
        self.last_action = action_to_take

        return action_to_take

    def compute_action_from_q_values(self, state: SimulationState, legal_actions: List[str]) -> str:
        actions = []
        for action in legal_actions:
            actions.append((action, self.get_q_value(state, action)))

        best_action = max(actions, key=lambda x: x[1])
        return best_action[0]

    def get_q_value(self, state: SimulationState, action: str) -> float:
        feats = self.extractor.get_features(state, action)

        return self.get_q_value_feats(feats)

    def get_q_value_feats(self, feats: Dict) -> float:
        q_val = 0

        for k in feats.keys():
            q_val += feats[k] * self.weights[k]
        return q_val

    def update(self, feats_action: Dict, next_state: SimulationState, reward: int) -> None:
        max_Q_sa_prim = self.compute_value_from_q_values(next_state)
        Q_sa = self.get_q_value_feats(feats_action)
        diff = reward + (self.gamma * max_Q_sa_prim) - Q_sa

        for k in feats_action.keys():
            self.weights[k] = self.weights[k] + (self.alpha * diff * feats_action[k])

        self.extractor.update_extractor(feats_action, next_state)
        self.lifepoints += reward

        if reward < 0:
            self.score += reward

    def compute_value_from_q_values(self, state: SimulationState) -> float:
        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        ghost_agents = state.ghost_agents
        legal_actions = state.grid.get_legal_actions(pos, ghost_agents)

        if not legal_actions:
            return 0.0

        values = []
        for action in legal_actions:
            values.append(self.get_q_value(state, action))

        return max(values)

    def get_exit(self, state: SimulationState) -> int:
        return self.get_closest_exit(state)[0]

    def get_policy(self, state: SimulationState) -> str:
        pos = state.schedule.agents_by_breed[type(self)][self.unique_id].pos
        ghost_agents = state.ghost_agents
        legal_actions = state.grid.get_legal_actions(pos, ghost_agents)

        return self.compute_action_from_q_values(state, legal_actions)

    def get_value(self, state) -> float:
        return self.compute_value_from_q_values(state)

    def get_weights(self) -> Dict:
        return self.weights

    def on_remove(self, state: SimulationState) -> Dict:
        guide = self.extractor.get_guide_obj(state)
        if guide.pos in state.grid.positions_by_breed[Exit]:
            last_reward = self.score * 2
        else:
            last_reward = self.score

        self.update(self.last_feats, state, last_reward)
        return {'epsilon': self.epsilon, 'alpha': self.alpha, 'gamma': self.gamma, 'weights': self.weights}

    def get_reward(self, feats: Dict, next_feats: Dict) -> float:

        if next_feats['uninformed_evacuees'] > 0:
            if feats['newly_informed_evacuees'] > 0:
                return int(9 * feats['newly_informed_evacuees'])
            else:
                return -1.0

        elif feats["closest_exit_distance"] > next_feats["closest_exit_distance"]:
            return 9.0
        else:
            return -1.0
