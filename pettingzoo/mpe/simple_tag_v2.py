from ._mpe_utils.simple_env import SimpleEnv, make_env
from .scenarios.simple_tag import Scenario
from PettingZoo.pettingzoo.utils.to_parallel import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, num_adversaries=3, num_good=1, num_neutral=2, num_obstacles=2, max_cycles=25):
        scenario = Scenario()
        world = scenario.make_world(num_adversaries=num_adversaries,
                                    num_good=num_good,
                                    num_neutral=num_neutral,
                                    num_obstacles=num_obstacles)
        super().__init__(scenario, world, max_cycles)


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
