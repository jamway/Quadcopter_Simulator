import numpy as np
from physics_sim import PhysicsSim
import time


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * len(self.sim.pose[2:]+[self.sim.time])
#         self.state_size = self.action_repeat * 1
        self.action_low = 300.
        self.action_high = 600.
        self.action_size = 4

        self.num_steps = 0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1. - .3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0.
        self.num_steps +=1
        pose_all = []
        for _ in range(self.action_repeat):
#             print(rotor_speeds)
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            self.done = done
            reward += self.get_reward()
            pose_all.append(self.sim.pose[2:]+[self.sim.time])
        next_state = np.concatenate(pose_all)

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.num_steps = 0
        state = np.concatenate([self.sim.pose[2:] + [self.sim.time]] * self.action_repeat)

#         state = np.array([self.sim.pose[2],self.sim.pose[2],self.sim.pose[2]])

        return state


class Hover(Task):
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        Task.__init__( self, init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
        self.target_pos[2] = target_pos[2] if target_pos[2] > 10. else 10.
        self.init_pose = init_pose
        self.runtime = runtime

        # Need to check current position and target position

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = np.tanh((1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()))
        reward = 0

        z_dist = abs(self.sim.pose[2] - self.target_pos[2])
#         rotation = abs(self.sim.pose[3:]%(2*np.pi)).sum()
        # reward -= .01 * abs(self.sim.pose[3:] - np.array([0.,0.,0.]).sum())
        # reward -= .05 * abs(self.sim.pose[3:]).sum()
        # reward -= 0.2 rotation/(2.0*np.pi)
#         reward += 10. - .2 * z_dist - abs(self.sim.pose[3:5]).sum()
        reward += 10. - .2 * z_dist - .1*abs(np.mod(self.sim.pose[3:5], 2*np.pi)).sum()
        if z_dist < 1.:
            reward += 30. - .5 * self.sim.v[2]
        # - .1 * (rotation/(2*nsp.pi)) if z_o_dist != 0 else 0.5 - .1 * (rotation/(2*np.pi))

#         if self.done and self.sim.time < self.sim.runtime:
#             reward -= 10


        return np.tanh(reward)
