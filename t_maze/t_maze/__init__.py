from gym.envs.registration import register
from t_maze.envs.t_maze import LOW, HIGH, CRASH_REWARD, FAIL_HOME

register(
        id='MiniGrid-TMaze-v0',
        entry_point='t_maze.envs:TMazeEnv'
)
register(
        id='MiniGrid-DoubleTMaze-v0',
        entry_point='t_maze.envs:DoubleTMazeEnv'
)

register(
        id='MiniGrid-TMazeHoming-v0',
        entry_point='t_maze.envs:TMazeHoming'
)