# Multiagent Evacuation Simulation

Main goal of the project is to create simulation of builidng evacuation in case of natural disaster. Evacuation in this
case is managed by group of Guides cooperating with each other to evacuate the most Evacuees possible in shortest
possible time.

## Quick launch instruction

To start simulation use one of two "main" files:

* main.py is simulation with visualisation in web browser
* main_no_graphics.py is simulation concentrated on learning guides AI algorithms, it does not support any graphics.

All parameters can be set using graphic interface or by editing dictionaries included in "main" files.

# Documentation

## General rules

* Agents can move one tile around them, both in + shape directions and x shape,
* Evacuees doesn't move, unless they were informed by Guide about exit,
* Guide and Evacuee share area is equal to one tile around them (+ and x)
*

## 1.) Simulation flow

1. Model object creation
2. Start of graphics server (if graphic application was chosen)
3. Model initialization (initialization of map, agents and all required variables)
4. Waiting for start command (or autostart in case of no-graphic mode)
5. Starting simulation loop from model object
    6. Model executes step() on embedded scheduler object
    7. Scheduler executes step() on every in the simulation, in fixed order
    8. Scheduler receives actions from agents and executes them both in Model and grid via Model object
    9. Terminal conditions of simulation are checked
10. Simulation finishes
11. New simulation is activated or results are saved to file system (in case of no-graphic mode)

## 2.) Modules

System consists of two modules.

### 2.1) "simulation" module

Module simulation is focused on running simulation itself, apart from agents logic. It consists on three major parts:

* Model - main object of simulation. Is responsible for maintaining simulation, it's states, adding and removing agents
  and every top-level thing about simulation.
    * EvacuationGrid - object maintaining positions of the guides in the simulation and providing tools
      position-related,
    * EvacuationSchedule - object maintaining behaviour of agents. It executes agents actions and store agents object.
      It also provides tools for agent management.

A little side part of this module is SimulationState object, which works as buffer between direct model variables and
agents. It basically contains all informations about simulation required by agents. It can be considered as "screenshot"
of one moment of the simulation.

### 2.2) "agents" module

Agents module is focused on agents definitions and their logics. It consists of three major parts:

* agents - file containing definition of all agents except guide agents. It also contains GuideAgent class, which is
  abstract and serves as a parent to all other Guide classes.
* agents_guides - file containing all guide agents definitions
* feature_extractor - serves as feature extractor (creates feature set from simulation state) for q learning guide
  agent. It also contains functions used for distance map generation, the ones used in multithread processing.

### 2.3) main files

These files are used to start simulation.

* main - this file can be used for starting simulation in graphic enabled mode. It consists of:
    * portrayal function which is used to define graphic representation of each agent
    * model_params dictionary which includes default model parameters and implements graphic object to modify them (such
      as Slider, Checkbox etc.)
* main_no_graphic - starts simulation in no grahpic mode. It's purpose is to train guide agents algorithms more
  efficiently, without loosing time on generating graphics.
    * model_params are slightly different than ones in main file. This one contains not visual objects but raw values
      used by model.
    * It also contains loop used for running simulation multiple times. Each time we run new simulation,
      we decrease epsilon value to improve learning with q learning algorithm. After loop finishes, learning results are
      saved to file.

Model params dictionary must contain the same keys for both simulation types.

## 3.) Classes by modules
### 3.1) "simulation" module
EvacuationModel

SimulationState

EvacuationGrid

EvacuationScheduler
### 3.2.1) "agents" module, agents.py file
#### StateAgent()

#### Obstacle(StateAgent)

#### Exit(StateAgent)

#### Sensor(StateAgent)

#### MapInfo(StateAgent)

#### GuideAgent(StateAgent)

#### Evacuee(StateAgent)

### 3.2.2) "agents" module, agents.py file
#### GuideABT(GuideAgent)

#### GuideQLearning(GuideAgent)

#### FeatureExtractor()