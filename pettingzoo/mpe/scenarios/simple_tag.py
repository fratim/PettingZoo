import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self,
                   num_adversaries=3,
                   num_good=1,
                   num_neutral=2,
                   num_obstacles=2,
                   abilities_goods=1,
                   abilities_adversaries=1,
                   abilities_neutrals=1,
                   obs_adv_speeds=True):
        world = World()

        world.obs_adversary_speeds = obs_adv_speeds
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_neutral = num_neutral
        num_agents = num_adversaries + num_good_agents + num_neutral
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            agent.good = True if num_adversaries <= i < (num_adversaries + num_good) else False
            agent.neutral = True if (num_adversaries + num_good) <= i else False

            if agent.adversary:
                base_name = "adversary"
                base_index = i
                agent.size = 0.075
                agent.accel = 3.0 * abilities_adversaries
                agent.max_speed = 1.0 * abilities_adversaries
            elif agent.good:
                base_name = "good"
                base_index = i - num_adversaries
                agent.size = 0.05
                agent.accel = 4.0 * abilities_goods
                agent.max_speed = 1.3 * abilities_goods
            elif agent.neutral:
                base_name = "neutral"
                base_index = i - num_adversaries - num_good
                agent.size = 0.075
                agent.accel = 4.0 * abilities_neutrals
                agent.max_speed = 1.3 * abilities_neutrals
            else:
                raise ValueError("Unknown Agent")


            agent.name = '{}_{}'.format(base_name, base_index)

            agent.collide = True
            agent.silent = True

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents

        n_adversary = 0
        n_good = 0
        n_neutral = 0

        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35 + 0.2 * n_adversary])
                n_adversary += 1
            elif agent.good:
                agent.color = np.array([0.35 + 0.2 * n_good, 0.85, 0.35])
                n_good += 1
            elif agent.neutral:
                agent.color = np.array([0.35, 0.35 + 0.2 * n_neutral, 0.85])
                n_neutral += 1

            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if (not agent.adversary and not agent.neutral)]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary:
            main_reward = self.adversary_reward(agent, world)
        elif agent.good:
            main_reward = self.agent_reward(agent, world)
        elif agent.neutral:
            main_reward = 0

        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            if other.neutral and not agent.neutral: # only friendly agent can observe friendly agents
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary or world.obs_adversary_speeds:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
