from gym_minigrid.minigrid import *
from gym_minigrid.register import register

LOW = 0.3
HIGH = 1

class MazeEnd(Ball):
    reward = LOW
    def __init__(self, reward='low'):
        color = 'yellow'
        if reward == 'high':
            color = 'green'
            self.reward = HIGH
        super().__init__(color)

    def can_overlap(self):
        return True

    def set_high(self):
        self.reward = HIGH

class TMazeEnv(MiniGridEnv):

    def __init__(self, highRewardEnd=1):
        size = 7
        self.LEFT_END = (1, size - 2)
        self.RIGHT_END = (size - 2, size - 2)
        self.middle = size // 2
        self.agent_start_pos = (self.middle,1)
        self.agent_start_dir = 1
        self.highRewardEnd = highRewardEnd
        super().__init__(
            grid_size= size,
            max_steps = 100,
            agent_view_size = 3
        )

        self.reward_range = (-1,1)


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                self.put_obj(Lava(),i,j)

        for i in range (width):
            self.grid.set(i,height-2,None)
            self.grid.set(self.middle,i,None)
        self.mission = "Find the highest reward"

        x=0
        y=0
        w=width
        h=height
        self.grid.horz_wall(x, y, w,Lava)
        self.grid.horz_wall(x, y + h - 1, w,Lava)
        self.grid.vert_wall(x, y, h,Lava)
        self.grid.vert_wall(x + w - 1, y, h,Lava)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.put_obj(MazeEnd('high'),self.LEFT_END[0], self.LEFT_END[1])
        self.put_obj(MazeEnd('low'),self.RIGHT_END[0], self.RIGHT_END[1])
        if self.highRewardEnd == 2:
            self.put_obj(MazeEnd('low'),self.LEFT_END[0], self.LEFT_END[1])
            self.put_obj(MazeEnd('high'),self.RIGHT_END[0], self.RIGHT_END[1])

    def switch_reward_pos(self):
        if self.highRewardEnd == 1:
            self.highRewardEnd = 2
            self.put_obj(MazeEnd('low'),self.LEFT_END[0], self.LEFT_END[1])
            self.put_obj(MazeEnd('high'),self.RIGHT_END[0], self.RIGHT_END[1])
        else:
            self.highRewardEnd = 1
            self.put_obj(MazeEnd('high'), self.LEFT_END[0], self.LEFT_END[1])
            self.put_obj(MazeEnd('low'), self.RIGHT_END[0], self.RIGHT_END[1])

if __name__ == "__main__":
    #Matplotlib crashes on render for now but it is not going to be a problem
    tmaze = TMazeEnv()
    tmaze.render()
    x = input("Press enter when finished")

register(
        id='MiniGrid-TMaze-v0',
        entry_point='t_maze:TMazeEnv'
)