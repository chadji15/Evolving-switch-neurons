Evolving-switch-neurons
========================
---
For this project, our aim is to incorporate the concept of the switch neurons in state-of-the-art evolutionary
algorithms in order to produce networks with switch neurons without the need to manually design them. 
We will judge the viability of such networks by testing them on simple reinforcement learning tasks such as
association tasks and T-maze tasks and comparing their results with those of neural networks designed by hand.

Install
---------
---
(needs python 3.7 and above)
>git clone https://github.com/chadji15/Evolving-switch-neurons.git  

->create a new python virtual environment
>cd Evolving-switch-neurons  
>pip install -r requirements.txt  

*Potentially needed*:  
>pip install -e gym_association_task  
>pip install -e t_maze  

Manual control:
--------------
---
* T-maze:
  >python manual_control.py --env MiniGrid-TMaze-v0  
* Double T-maze: 
  >python manual_control.py --env MiniGrid-DoubleTMaze-v0  
* Association Task: 
  >cd gym_association_task/gym_association_task/envs  
  python association_task_env.py  
  
Run some experiments
---------------------
---
<h3>NEAT:</h3>  

To run a NEAT experiment with switch neurons, the main file used is evolve.py.  
Execution:  
`python evolve.py -s <scheme> -c <config> -g <generations> -p <problem>`  
Where `<scheme>` is one of `switch` or `switch_maps`. There is also an option for `recurrent` but it
hasn't been tested much.  
`<config>` is the path to the config file for NEAT. A config file defines the various probabilities
and value ranges for the genome parameters. Config files are present in the repository under /config.
for more information on configuration files refer to 
<https://neat-python.readthedocs.io/en/latest/config_file.html>  
`<generations>` is the number of maximum generations allowed.  
`<problem>` is one of `binary_association` (3x3 one-to-one), `tmaze` or `double_tmaze`.

More options are availabe. You can see them with `python evolve.py`.
Example:
>python evolve.py -p tmaze -s switch -c config/config-switch-tmaze

For NEAT experiments with the guided map-based encoding , one needs to execute the guided_maps.py:  
`python guided_maps.py <config> <generations> <resdir> <taskno>`  
Where `<config>` and `<generations>` server the same purpose.  
`<resdir>` is a directory where the results will be stored (a binary file with the winning genome,
a statistics plot of the maximum fitness and the network visualization).  
`<taskno>` is used when executing this in batch mode. Used to separate the result files. Can be set
to any number with no consequence if only one instance is running.

Currently, the evaluated problem is the skinner 3x10 one-to-one but this
can change with a simple change of the evaluation and creation function.

Example:
>python guided_maps.py config/binary-guided-maps 500 out 0

<h3>Quality Diversity:</h3>
The main file used for quality diversity experiments is deap_neat.py  
`python deap_neat.py -p <problem> -c <config> -hp <params>`   
`<problem>` is one of `skinner2`, `skinner3`, `skinner4`, `skinner3x10`, `tmaze` although tmaze is
not tested much.  
`<config>` is the path to the NEAT config file.
`<params>` is the path to a second configuration file that defines the parameters of the problem
and the QD algorithm. Such parameter files are provided under /params.  
*Note*: Mutation probability is always set to 1 due to the fact that 
mutation probabilities are taken from the NEAT config file.

Example:
>python deap_neat.py -p skinner3 -c config/deap-skinner3 -hp params/skinner3.yaml

---
Further documentation that could be helpful to a programmer
can be found in each individual file's comments.