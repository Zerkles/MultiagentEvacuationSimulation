import random
import time
from statistics import mean

from matplotlib import pyplot as plt

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

    "cross_gap": 10,
    "boxes_thickness": 15,
    "rectangles_num": 10,
    "rectangles_max_size": 15,
    "erosion_proba": 0.5,
}
# TODO: Change default FPS to 0 (Max possible)

nums_evacuees = [100, 200, 500, 1000, 1500, 2000]

n_repeats = 5
results = []

for num in nums_evacuees:

    partial_results = []
    for _ in range(n_repeats):
        model_params["evacuees_num"] = num
        model = EvacuationModel(**model_params)
        model.reset_randomizer()
        partial_results.append(model.run_model())

    results.append(mean(partial_results))

plt.plot(nums_evacuees, results)
plt.title("Time of simulation for different number of evacuees")
plt.xlabel("Initial number of evacuees")
plt.ylabel("Time [s]")
plt.savefig("no_graphic.png")
plt.show()
print(results)
