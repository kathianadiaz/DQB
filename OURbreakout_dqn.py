import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
EPISODES = 50000


class DQNAgent:
  def __init__(self, action_size):
      self.render = False
      self.load_model = False
      self.state_size = (84, 84, 4)
      self.action_size = 3
      self.epsilon = 1.
      self.epsilon_start, self.epsilon_end = 1.0 , 0.10
      self.exploration_steps = 1000000
      self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
      self.learning_rate = 0.10 
      self.batch_size = 16
      self.train_start = 50000
      self.update_target_rate = 10000
      self.discount_factor = 0.90
      self.memory = deque(maxlen=400000)
      self.no_op_steps = 30
      self.model = self.build_model()
      self.target_model = self.build_model()
      self.update_target_model()
      self.optimizer = self.optimizer()
      self.sess = tf.InteractiveSession()
      K.set_session(self.sess)
      self.avg_q_max, self.avg_loss = 0, 0
      self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
      self.summary_writer = tf.summary.FileWriter('summary/breakout_dqn', self.sess.graph)
      self.sess.run(tf.global_variables_initializer())
      if self.load_model:
          self.model.load_weights('./save_model/breakout_dqn.h5')

  def build_model(self):
      model = Sequential()
      model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',input_shape=self.state_size))
      model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
      model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
      model.add(Flatten())
      model.add(Dense(512, activation='relu'))
      model.add(Dense(self.action_size))
      return model

  def optimizer(self):
      a = K.placeholder(shape=(None,), dtype='int32')
      y = K.placeholder(shape=(None,), dtype='float32')

      py_x = self.model.output
      a_one_hot = K.one_hot(a, self.action_size)
      q_value = K.sum(py_x * a_one_hot, axis=1)
      error = K.abs(y - q_value)
      quadratic_part = K.clip(error, 0.0, 1.0)
      linear_part = error - quadratic_part
      loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
      optimizer = RMSprop(lr =0.00025, epsilon=0.01)
      updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
      train = K.function([self.model.input, a, y], [loss], updates=updates)

      return train

  def update_target_model(self):
      self.target_model.set_weights(self.model.get_weights())

  def get_action(self, history):
      history = np.float32(history / 255.0)
      if np.random.rand() <= self.epsilon:
          return random.randrange(self.action_size)
      else:
          q_value = self.model.predict(history)
          return np.argmax(q_value[0])

  def replay_memory(self, history, action, reward, next_history, dead):
      self.memory.append((history, action, reward, next_history, dead))

  def train_replay(self):
      if len(self.memory) < self.train_start:
          return
      if self.epsilon > self.epsilon_end:
          self.epsilon -= self.epsilon_decay_step
      mini_batch = random.sample(self.memory, self.batch_size)
      history = np.zeros((self.batch_size, self.state_size[0],self.state_size[1], self.state_size[2]))
      next_history = np.zeros((self.batch_size, self.state_size[0],self.state_size[1], self.state_size[2]))
      target = np.zeros((self.batch_size,))
      action, reward, dead = [], [], []
      for i in range(self.batch_size):
          history[i] = np.float32(mini_batch[i][0] / 255.)
          next_history[i] = np.float32(mini_batch[i][3] / 255.)
          action.append(mini_batch[i][1])
          reward.append(mini_batch[i][2])
          dead.append(mini_batch[i][4])
      target_value = self.target_model.predict(next_history)

      for i in range(self.batch_size):
          if dead[i]:
              target[i] = reward[i]
          else:
              target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])
      loss = self.optimizer([history, action, target])
      self.avg_loss += loss[0]

  def save_model(self, name):
      self.model.save_weights(name)

  def setup_summary(self):
      episode_total_reward = tf.Variable(0.)
      episode_avg_max_q = tf.Variable(0.)
      episode_duration = tf.Variable(0.)
      episode_avg_loss = tf.Variable(0.)
      tf.summary.scalar('Total Reward/Episode', episode_total_reward)
      tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
      tf.summary.scalar('Duration/Episode', episode_duration)
      tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
      summary_vars = [episode_total_reward, episode_avg_max_q,episode_duration, episode_avg_loss]
      summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
      update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
      summary_op = tf.summary.merge_all()
      return summary_placeholders, update_ops, summary_op

def pre_processing(observe):
  processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
  return processed_observe

def run():
  env = gym.make("BreakoutDeterministic-v0")
  agent = DQNAgent(action_size=3) # 3
  scores, episodes, global_step = [], [], 0
  for e in range(EPISODES):
      done = False
      dead = False
      step, score, start_life = 0, 0, 5

      observe = env.reset()
      for _ in range(random.randint(1, agent.no_op_steps)):
          observe, _, _, _ = env.step(1)

      state = pre_processing(observe)
      history = np.stack((state, state, state, state), axis=2)
      history = np.reshape([history], (1, 84, 84, 4))
      while not done:
          #env.render()
          global_step += 1
          step += 1
          action = agent.get_action(history)
          if action == 0:
              real_action = 1
          elif action == 1:
              real_action = 2
          else:
              real_action = 3
          observe, reward, done, info = env.step(real_action)
          next_state = pre_processing(observe)
          next_state = np.reshape([next_state], (1, 84, 84, 1))
          next_history = np.append(next_state, history[:, :, :, :3], axis=3)
          agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])
          if start_life > info["ale.lives"]:
              dead = True
              start_life = info["ale.lives"]
