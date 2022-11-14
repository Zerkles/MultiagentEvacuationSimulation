import json
import os
from typing import Dict

from simulation.model import EvacuationModel

HEIGHT = WIDTH = 100

model_params = {
    "width": WIDTH,
    "height": HEIGHT,

    "ghost_agents": False,
    "show_map": False,

    "guides_num": 2,
    "guides_mode": "Q Learning",
    "guides_random_position": False,

    "evacuees_num": 500,
    "evacuees_share_information": False,

    "map_type": 'default',

    "cross_gap": 10,
    "boxes_thickness": 15,
    "rectangles_num": 10,
    "rectangles_max_size": 15,
    "erosion_proba": 0.5,

    "qlearning_params": {'epsilon': 0.8, 'gamma': 0.8, 'alpha': 0.2, 'weights': None}
}

qlearning_params = model_params['qlearning_params']

n_games = 50
epsilon_min = 0.2
epsilon_diff = (qlearning_params['epsilon'] - epsilon_min) / n_games

for i in range(n_games):
    print(i, qlearning_params)

    model = EvacuationModel(**model_params)
    model.run_model()
    model.reset_randomizer()

    qlearning_params = model.qlearning_params
    print(model.qlearning_params)

    if qlearning_params['epsilon'] >= epsilon_min:
        qlearning_params['epsilon'] -= epsilon_diff

    model_params['qlearning_params'] = qlearning_params

with open(f"output/weights.txt", "w") as f:
    f.write(json.dumps(qlearning_params['weights']))
