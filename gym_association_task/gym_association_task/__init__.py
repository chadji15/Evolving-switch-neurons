from gym.envs.registration import register

register(
    id='OneToOne2x2-v0',
    entry_point='gym_association_task.envs:OneToOne2x2',
)

register(
    id='OneToOne3x3-v0',
    entry_point='gym_association_task.envs:OneToOne3x3',
)

register(
    id='OneToOne4x4-v0',
    entry_point='gym_association_task.envs:OneToOne4x4',
)

register(
    id='OneToMany3x2-v0',
    entry_point='gym_association_task.envs:OneToMany3x2',
)

register(
    id='ManyToMany2x2Rand-v0',
    entry_point='gym_association_task.envs:ManyToMany2x2Rand',
)

register(
    id='OneToOne3x10-v0',
    entry_point='gym_association_task.envs:OneToOne3x10',
)
