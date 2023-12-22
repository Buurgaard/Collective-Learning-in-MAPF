class R:
    STEP = -1
    GOAL = 100
    COLLISION = -10
    OBSTACLE = -5
    WAIT = -2


class Net:
    BATCH_SIZE = 256
    GAMMA = 0.75
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.001
    LR = 1e-4
    MEM_SIZE = 5000
    FOV = 3
    EXTENDED_FOV = 5
    DROP_OUT = True
    DROP_PROB = 0.05


class Com:
    ERM_SEND = 20
    RANGE = 2
    GOAL_ONLY = False
    All = True
    CONFLICT = False
    GOAL = False