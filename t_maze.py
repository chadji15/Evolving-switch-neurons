from gym_minigrid.minigrid import *
from gym_minigrid.register import register

LOW = 0.2
HIGH = 1
CRASH_REWARD = -0.4
FAIL_HOME = -0.3

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

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, highRewardEnd=1):
        size = 7
        self.actions = TMazeEnv.Actions
        self.LEFT_END = (1, size - 2)
        self.RIGHT_END = (size - 2, size - 2)
        self.middle = size // 2
        self.TURNING_POINTS = [(self.middle, size - 2)]
        self.agent_start_pos = (self.middle,1)
        self.agent_start_dir = 1
        self.highRewardEnd = highRewardEnd
        super().__init__(
            grid_size= size,
            max_steps = 17,
            agent_view_size = 3
        )

        self.reward_range = (-1,1)


    def _gen_grid(self, width, height):
        self.reward = 0
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

        self.put_obj(Goal(),self.agent_start_pos[0], self.agent_start_pos[1])

    def switch_reward_pos(self):
        if self.highRewardEnd == 1:
            self.highRewardEnd = 2
            self.put_obj(MazeEnd('low'),self.LEFT_END[0], self.LEFT_END[1])
            self.put_obj(MazeEnd('high'),self.RIGHT_END[0], self.RIGHT_END[1])
        else:
            self.highRewardEnd = 1
            self.put_obj(MazeEnd('high'), self.LEFT_END[0], self.LEFT_END[1])
            self.put_obj(MazeEnd('low'), self.RIGHT_END[0], self.RIGHT_END[1])


    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
                reward = CRASH_REWARD
            if fwd_cell != None and fwd_cell.type == 'ball':
                self.reward = fwd_cell.reward
                self.grid.set(self.agent_pos[0],self.agent_pos[1],None)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True
            reward = FAIL_HOME

        atHome = list(self.agent_pos) == list(self.agent_start_pos)
        atTurning = False
        for turningPoint in self.TURNING_POINTS:
            if list(turningPoint) == list(self.agent_pos):
                atTurning = True
        atMazeEnd = list(self.agent_pos) == list(self.LEFT_END) or list(self.agent_pos) == list(self.RIGHT_END)

        #observation format = [is agent at home, is agent at turning point, is agent at maze end]
        obs = (float(atHome),float(atTurning),float(atMazeEnd))
        print(obs)
        return obs, reward, done, {}

    def _reward(self):
        return self.reward


if __name__ == "__main__":
    #Matplotlib crashes on render for now but it is not going to be a problem
    tmaze = TMazeEnv()
    tmaze.render()
    x = input("Press enter when finished")

register(
        id='MiniGrid-TMaze-v0',
        entry_point='t_maze:TMazeEnv'
)