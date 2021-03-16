import ma_gym as _
import unittest

from env_wrapper import VectorizedMultiAgentEnvWrapper
from gym import spaces
from torch import rand as torch_rand
from torch import round as torch_round
from torch import Tensor

class Test_ActionConverter(unittest.TestCase):
    def test_pong_duel_converter(self):
        mapper = lambda actions: torch_round(actions * 1.5 + 1).flatten().tolist()
        action_tensor = torch_rand((2,)) * 2 -1
        action_list   = mapper(action_tensor)
        self.assertEqual(len(action_list), 2, "the action list is not of dim 2")
        self.assertTrue(all([0 <= i <= 2 for i in action_list]), "the action list is not mapped to [0, 2]")
        print("Origin:", action_tensor)
        print("Action:", action_list)
        

class Test_WrappingPongDuel(unittest.TestCase):
    def test_pong_duel_wrap(self):
        wrapper = VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper(
            env="PongDuel-v0", 
            mapper=lambda actions: torch_round(actions * 1.5 + 1).flatten().tolist()
        )
        self.assertEqual(wrapper.observation_space.shape, (2,10))
        self.assertEqual(wrapper.action_space.shape,      (2,))


    def test_agent_wrap_act(self):
        wrapper = VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper(
            env="PongDuel-v0", 
            mapper=lambda actions: torch_round(actions * 1.5 + 1).flatten().tolist()
        )
        # case one: discrete
        wrapper.action_space_ = spaces.Discrete(3)
        wrapper.agent_num     = 4
        act = wrapper._wrap_act()
        self.assertTrue(isinstance(act, spaces.Box), "[4 agents, discrete(3)] action space is not gym.spaces.Box")
        self.assertEqual(act.shape, (4,), "[4 agents, discrete(3)] action shape is not correct: " + str(act.shape))
        # case two: box
        wrapper.action_space_ = spaces.Box(low=-1.0, high=1.0, shape=(5,7))
        wrapper.agent_num     = 3
        act = wrapper._wrap_act()
        self.assertTrue(isinstance(act, spaces.Box), "[3 agents, box(5,7)] action space is not gym.spaces.Box")
        self.assertEqual(act.shape, (3, 5, 7), "[3 agents, box(5,7)] action shape is not correct: " + str(act.shape))
        # case three: list
        wrapper.agent_num     = 5
        wrapper.action_space_ = [
            spaces.Box(low=-1.0, high=1.0, shape=(1,7)),
            spaces.Box(low=-1.0, high=1.0, shape=(1,7))
        ]
        act = wrapper._wrap_act()
        self.assertTrue(isinstance(act, spaces.Box), "[5 agents, [2*box(1,7)]] action space is not gym.spaces.Box")
        self.assertEqual(act.shape, (5, 2, 1, 7), "[5 agents, [2*box(1,7)]] action shape is not correct: " + str(act.shape))
        # case three: list
        wrapper.agent_num     = 5
        wrapper.action_space_ = [
            spaces.Discrete(3),
            spaces.Discrete(3),
            spaces.Discrete(3),
            spaces.Discrete(3)
        ]
        act = wrapper._wrap_act()
        self.assertTrue(isinstance(act, spaces.Box), "[5 agents, [4*Discrete(4)]] action space is not gym.spaces.Box")
        self.assertEqual(act.shape, (5, 4), "[5 agents, [4*Discrete(4)]] action shape is not correct: " + str(act.shape))
        
    # def test_vectorized_multiagent_env_wrapper(self):
    #     from torch import round as torch_round
    #     wrapper = VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper(
    #         env="PongDuel-v0", 
    #         mapper=lambda actions: torch_round(actions * 1.5 + 1).flatten().tolist()
    #     )
    #     self.assertTrue(check_env(wrapper), "env check for wrapper does not pass")
        
    def test_agent_step(self):
        wrapper = VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper(
            env="PongDuel-v0", 
            mapper=lambda actions: torch_round(actions * 1.5 + 1).flatten().tolist()
        )
        wrapper.reset()
        mock_action = torch_rand(size=(2,))
        states, reward, done, _ = wrapper.step(mock_action)
        self.assertTrue(isinstance(states, Tensor))
        self.assertTrue(isinstance(reward, Tensor))
        self.assertTrue(isinstance(done, list))
        self.assertEqual(states.shape, (2,10))
        self.assertEqual(reward.shape, (1,2))
        
        
if __name__ == '__main__':
    unittest.main()