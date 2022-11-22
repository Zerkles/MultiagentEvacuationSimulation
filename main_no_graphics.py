import json
import os
from multiprocessing import set_start_method
from typing import Dict

from agents.feature_extractor import FeatureExtractor
from simulation.grid import EvacuationGrid
from simulation.model import EvacuationModel, get_feature_extractor_maps

HEIGHT = WIDTH = 100

model_params = {
    "width": WIDTH,
    "height": HEIGHT,

    "ghost_agents": False,
    "show_map": False,

    "guides_num": 2,
    "guides_mode": "Q Learning",
    "guides_random_position": False,

    "evacuees_num": 1000,
    "evacuees_share_information": True,

    "map_type": 'default',

    "cross_gap": 10,
    "boxes_thickness": 15,
    "rectangles_num": 10,
    "rectangles_max_size": 15,
    "erosion_proba": 0.5,

    "qlearning_params": {'epsilon': 0.8, 'gamma': 0.8, 'alpha': 0.2, 'weights': None},
    "extractor_maps": None,
}

if __name__ == '__main__':
    qlearning_params = model_params['qlearning_params']

    n_games = 2000
    epsilon_min = 0.2
    epsilon_diff = (qlearning_params['epsilon'] - epsilon_min) / n_games

    for i in range(n_games):
        model = EvacuationModel(**model_params)
        model.run_model()
        model.reset_randomizer()

        print(f"it: {i}; steps: {model.schedule.steps}; qlearning_params: {qlearning_params}")

        qlearning_params = model.qlearning_params
        model_params["extractor_maps"] = FeatureExtractor.maps

        if qlearning_params['epsilon'] >= epsilon_min:
            qlearning_params['epsilon'] -= epsilon_diff

        model_params['qlearning_params'] = qlearning_params

    with open(f"output/weights.txt", "w") as f:
        f.write(json.dumps(qlearning_params['weights']))
