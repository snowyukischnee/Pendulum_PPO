import tensorflow as tf
import numpy as np
import gym
from typing import Any
from collections import deque
import random
import matplotlib.pyplot as plt
import sys

GAMMA = 0.9
EPSILON = 0.2
VALUE_COEF = 4e-1
DISTRIBUTION_COEF = 1e-3
ACTOR_LR = 1e-4
CRITIC_LR = 2e-4
TOTAL_TRAINING_EPISODES = 1000
MAX_STEPS = 200
MEMORY_SIZE = 32
BATCH_SIZE = 32
UPDATE_INTERVAL = 32
N_ACTOR_UPDATE = 10
N_CRITIC_UPDATE = 10


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience: Any) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


class ActorCritic(object):
    def __init__(self, sess: tf.Session, state_size: int,
                 action_size: int, action_bound: np.ndarray,
                 actor_leanring_rate: float, critic_learning_rate: float) -> None:

        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.actor_leanring_rate = actor_leanring_rate
        self.critic_learning_rate = critic_learning_rate

        self.state = tf.placeholder(tf.float32, [None, self.state_size], 'state')
        self.action = tf.placeholder(tf.float32, [None, self.action_size], 'action')
        self.next_state = tf.placeholder(tf.float32, [None, self.state_size], 'next_state')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.v_next = tf.placeholder(tf.float32, [None, 1], 'v_next')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('Critic'):
            self.c_params, self.c_v = self.build_critic_network(self.state, 'Critic', 'critic_network')
            with tf.variable_scope('output'):
                self.c_advantage = self.reward + GAMMA * self.v_next - self.c_v
            with tf.variable_scope('loss'):
                self.c_loss = tf.reduce_mean(tf.square(self.c_advantage))
                self.c_optimizer = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.c_loss)
        with tf.variable_scope('Actor'):
            self.a_params, self.a_normdist, self.a_mean = self.build_actor_network(self.state, 'Actor', 'actor_current_pi')
            self.a_old_params, self.a_old_normdist, self.a_old_mean = self.build_actor_network(self.state, 'Actor', 'actor_old_pi', False)
            with tf.variable_scope('output'):
                self.action_prediction = tf.squeeze(self.a_normdist.sample(1), axis=0)
                self.action_play = tf.squeeze(self.a_mean, axis=0)
            with tf.variable_scope('loss'):
                a_ratio = self.a_normdist.prob(self.action) / self.a_old_normdist.prob(self.action)
                self.a_loss = tf.minimum(a_ratio * self.advantage, tf.clip_by_value(a_ratio, 1.0-EPSILON, 1.0+EPSILON) * self.advantage)
                self.a_loss -= VALUE_COEF * self.c_loss
                self.a_loss += DISTRIBUTION_COEF * self.a_normdist.entropy()
                self.a_loss = -tf.reduce_mean(self.a_loss)
                self.a_optimizer = tf.train.AdamOptimizer(self.actor_leanring_rate).minimize(self.a_loss)

    def update_network(self, origin: Any, target: Any) -> None:
        self.sess.run([t.assign(o) for o, t in zip(origin, target)])

    def build_actor_network(self, input_tensor: Any, outer_scope: str, name: str, trainable: bool = True) -> Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            with tf.variable_scope('feature_extract'):
                l1 = tf.layers.dense(
                    inputs=input_tensor,
                    units=100,
                    activation=tf.nn.relu,
                    name='l1',
                    trainable=trainable,
                )
            with tf.variable_scope('action_distribution'):
                mean = ((self.action_bound[1] - self.action_bound[0]) / 2) * tf.layers.dense(
                    inputs=l1,
                    units=self.action_size,
                    activation=tf.nn.tanh,
                    name='mean',
                    trainable=trainable,
                ) + (action_bound[1] + action_bound[0]) / 2
                variance = tf.layers.dense(
                    inputs=l1,
                    units=self.action_size,
                    activation=tf.nn.softplus,
                    name='variance',
                    trainable=trainable,
                )
                norm_dist = tf.distributions.Normal(loc=mean, scale=variance)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, norm_dist, mean

    def build_critic_network(self, input_tensor: Any, outer_scope: str, name: str, trainable: bool = True) ->  Any:
        if outer_scope and outer_scope.strip():
            full_path = outer_scope + '/' + name
        else:
            full_path = name
        with tf.variable_scope(name):
            with tf.variable_scope('feature_extract'):
                l1 = tf.layers.dense(
                    inputs=input_tensor,
                    units=100,
                    activation=tf.nn.relu,
                    name='l1',
                    trainable=trainable,
                )
            with tf.variable_scope('value'):
                v = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=None,
                    name='value',
                    trainable=trainable
                )
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_path)
        return params, v

    def learn(self, experiences: Any) -> None:
        state, action, next_state, reward = zip(*experiences)
        state = np.vstack(state)
        action = np.vstack(action)
        next_state = np.vstack(next_state)
        reward = np.vstack(reward)
        self.update_network(self.a_params, self.a_old_params)
        feed_dict = {
            self.state: next_state,
        }
        v_next = self.sess.run(self.c_v, feed_dict=feed_dict)
        feed_dict = {
            self.state: state,
            self.reward: reward,
            self.v_next: v_next
        }
        advantage = self.sess.run(self.c_advantage, feed_dict=feed_dict)
        feed_dict = {
            self.state: state,
            self.action: action,
            self.advantage: advantage,
            self.reward: reward,
            self.v_next: v_next
        }
        for _ in range(N_ACTOR_UPDATE):
            self.sess.run(self.a_optimizer, feed_dict=feed_dict)
        for _ in range(N_CRITIC_UPDATE):
            self.sess.run(self.c_optimizer, feed_dict=feed_dict)

    def predict(self, state: np.ndarray) -> Any:
        feed_dict = {self.state: state[np.newaxis, :]}
        action = self.sess.run(self.action_prediction, feed_dict=feed_dict)[0]
        return np.clip(action, self.action_bound[0], self.action_bound[1])

    def play(self, state: np.ndarray) -> Any:
        feed_dict = {self.state: state[np.newaxis, :]}
        action = self.sess.run(self.action_play, feed_dict=feed_dict)[0]
        return np.clip(action, self.action_bound[0], self.action_bound[1])


if __name__ == '__main__':
    env = gym.make('Pendulum-v0').unwrapped
    state_size = env.observation_space.shape[0]
    action_bound = np.array([env.action_space.low, env.action_space.high])
    action_size = 1
    sess = tf.Session()
    ac = ActorCritic(sess, state_size, action_size, action_bound, ACTOR_LR, CRITIC_LR)
    memory = Memory(MEMORY_SIZE)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./log/', sess.graph)
    pretrained = True
    training = False
    if sys.argv[1] == 'play':
        training = False
    else:
        training = True
    if pretrained:
        saver.restore(sess, "./models/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    global rewards
    if training is False:
        print('Begin testing')
        rewards = []
        for episode in range(100):
            total_reward = 0
            state = env.reset()
            t = 0
            while True:
                t += 1
                # env.render()
                action = ac.predict(state)
                next_state, reward, _, _ = env.step(action)
                total_reward += reward
                if t >= MAX_STEPS:
                    print('episode: {} '.format(episode), 'reward: {} '.format(total_reward))
                    rewards.append(total_reward)
                    break
                state = next_state
        print('Mean reward: {}'.format(np.mean(rewards)))
        print('Begin playing')
        state = env.reset()
        while True:
            env.render()
            action = ac.predict(state)
            next_state, reward, _, _ = env.step(action)
            state = next_state
    else:
        print('Begin Training')
        rewards = []
        update_t = 0
        total_reward = 0
        for episode in range(1, TOTAL_TRAINING_EPISODES + 1):
            t = 0
            state = env.reset()
            total_reward = 0
            while True:
                t += 1
                update_t += 1
                action = ac.predict(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                reward = (reward + 8) / 10
                memory.add((state, action, next_state, reward))
                if done or t > MAX_STEPS or update_t > UPDATE_INTERVAL:
                    update_t = 0
                    ac.learn(memory.sample(BATCH_SIZE, continuous=True))
                if done or t > MAX_STEPS:
                    memory.clear()
                    print('episode: {} '.format(episode), 'reward: {} '.format(total_reward))
                    rewards.append(total_reward)
                    break
                state = next_state
            if episode % 5 == 0:
                save_path = saver.save(sess, './models/model.ckpt')
                print('model Saved at {}'.format(save_path))
        print('Mean reward: {}'.format(np.mean(rewards)))
        plt.plot(np.squeeze(rewards))
        plt.show()