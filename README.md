# Evolving-switch-neurons
For this project, our aim is to incorporate the concept of the switch neurons in state-of-the-art evolutionary
algorithms in order to produce networks with switch neurons without the need to manually design them. 
We will judge the viability of such networks by testing them on simple reinforcement learning tasks such as
association tasks and T-maze tasks and comparing their results with those of neural networks designed by hand.

Manual control:

* T-maze: python manual_control.py --env MiniGrid-TMaze-v0

* Double T-maze: python manual_control.py --env MiniGrid-DoubleTMaze-v0

* Association Task: cd gym_association_task/gym_association_task/envs\
python association_task_env.py