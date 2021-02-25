from enum import Enum

class td3_parameters(Enum):
    LEARNING_RATE       = 0.001
    BUFFER_SIZE         = 100000
    LEARNING_STARTS     = 100
    BATCH_SIZE          = 100
    TAU                 = 0.005
    GAMMA               = 0.99
    GRADIENT_STEPS      = -1 #DISABLE
    N_EPISODES_ROLLOUT  = 1
    POLICY_DELAY        = 2

class training_paramters(Enum):
    MULTI_AGENT_ENV_NAME = "PongDuel-v0"
    DEFAULT_LEARN_STEPS  = 10000
    DEFAULT_TEST_STEPS   = 10000
