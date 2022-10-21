import time
from mesa.visualization.UserParam import Slider, Checkbox, Choice, StaticText, UserSettableParameter, UserParam

from model import EvacuationModel

HEIGHT = WIDTH = 100

model_params = {
    "width": WIDTH,
    "height": HEIGHT,

    "ghost_agents": False,
    "show_map": False,

    "guides_num": 2,
    "guides_mode": "none",
    "guides_random_position": False,

    "evacuees_num": 2000,
    "evacuees_share_information": False,

    "map_type": 'default',

    "rectangles_num": 10,
    "rectangles_max_size": 15,
    "erosion_proba": 0.5,
}

model = EvacuationModel(**model_params)
start_time = time.time()
model.run_model()
print("execution time:", time.time() - start_time, " [s]")

# TODO: Change default FPS to 0 (Max possible)
