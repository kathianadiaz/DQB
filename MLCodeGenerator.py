algorithm = " "
code = []


def set_algorithm(alg):
    global algorithm
    algorithm = alg

def ini():
    if algorithm == 'PG':
         py = "import gym\n" + \
              "import numpy as np\n" + \
              "from keras.models import Sequential\n" + \
              "from keras.layers import Dense, Reshape, Flatten\n" + \
              "from keras.optimizers import Adam\n" + \
              "from keras.layers.convolutional import Convolution2D\n" + \
              "\n"+ \
              "\n" + \
              "class PGAgent:\n"
         code.append(py)

    elif algorithm == "QL":
        py = "import gym\n" + \
             "import random\n" + \
             "import numpy as np\n" + \
             "import tensorflow as tf\n" + \
             "from collections import deque\n" + \
             "from skimage.color import rgb2gray\n" + \
             "from skimage.transform import resize\n" + \
             "from keras.models import Sequential\n" + \
             "from keras.optimizers import RMSprop\n" + \
             "from keras.layers import Dense, Flatten\n" + \
             "from keras.layers.convolutional import Conv2D\n" + \
             "from keras import backend as K\n" + \
             "EPISODES = 50000\n" + \
             "\n" + \
             "\n" + \
             "class DQNAgent:\n"
        code.append(py)

def model_parametersPG(learning_rate, discount_factor):
    py = "  def __init__(self, state_size, action_size):\n" + \
         "      self.state_size = state_size\n" + \
         "      self.action_size = action_size\n" + \
         "      self.gamma = "+ discount_factor + '\n' + \
         "      self.learning_rate = "+ learning_rate + '\n' + \
         "      self.states = []\n" + \
         "      self.gradients = []\n" + \
         "      self.rewards = []\n" + \
         "      self.probs = []\n" + \
         "      self.model = self._build_model()\n" + \
         "      self.model.summary()\n"

    code.append(py)

def model_parametersQL(Learning_Rate,Epsilon_Start,Epsilon_End,Exploration_Steps,Batch_Size,Discount_Factor,No_Op_Steps,Action_Size):
    py = "  def __init__(self, action_size):\n" + \
         "      self.render = False\n" + \
         "      self.load_model = False\n" + \
         "      self.state_size = (84, 84, 4)\n" + \
         "      self.action_size = " + Action_Size + "\n" + \
         "      self.epsilon = 1.\n" + \
         "      self.epsilon_start, self.epsilon_end = " + Epsilon_Start + " , " + Epsilon_End + "\n" + \
         "      self.exploration_steps = " + Exploration_Steps + "\n" + \
         "      self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps\n" + \
         "      self.learning_rate = " + Learning_Rate + " \n" + \
         "      self.batch_size = " + Batch_Size + "\n" + \
         "      self.train_start = 50000\n" + \
         "      self.update_target_rate = 10000\n" + \
         "      self.discount_factor = " + Discount_Factor + "\n" + \
         "      self.memory = deque(maxlen=400000)\n" + \
         "      self.no_op_steps = " +No_Op_Steps + "\n" + \
         "      self.model = self.build_model()\n" + \
         "      self.target_model = self.build_model()\n" + \
         "      self.update_target_model()\n" + \
         "      self.optimizer = self.optimizer()\n" + \
         "      self.sess = tf.InteractiveSession()\n" + \
         "      K.set_session(self.sess)\n" + \
         "      self.avg_q_max, self.avg_loss = 0, 0\n" + \
         "      self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()\n" + \
         "      self.summary_writer = tf.summary.FileWriter('summary/breakout_dqn', self.sess.graph)\n" + \
         "      self.sess.run(tf.global_variables_initializer())\n" + \
         "      if self.load_model:\n" + \
         "          self.model.load_weights('./save_model/breakout_dqn.h5')\n\n"

    code.append(py)


def ConvLayers():
    py=" "
    if algorithm == "PG":
        py = "  def _build_model(self):\n" + \
             "      model = Sequential()\n" + \
             "      model.add(Reshape((1, 80, 80), input_shape=(self.state_size,))) \n" + \
             "      model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same', activation='relu', init='he_uniform')) \n" + \
             "      model.add(Flatten())\n"
        code.append(py)

    if algorithm == "QL":
        py = "  def build_model(self):\n" + \
             "      model = Sequential()\n" + \
             "      model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',input_shape=self.state_size))\n" + \
             "      model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))\n" + \
             "      model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))\n" + \
             "      model.add(Flatten())\n"
        code.append(py)



def PredLayers():

    if algorithm == "PG":
        py = "      model.add(Dense(64, activation='relu', init='he_uniform'))\n" + \
             "      model.add(Dense(32, activation='relu', init='he_uniform'))\n" + \
             "      model.add(Dense(self.action_size, activation='softmax'))\n" + \
             "      opt = Adam(lr=self.learning_rate)\n" + \
             "      model.compile(loss='categorical_crossentropy', optimizer=opt)\n" + \
             "      return model\n\n"
        code.append(py)

    if algorithm == "QL":
        py = "      model.add(Dense(512, activation='relu'))\n" + \
             "      model.add(Dense(self.action_size))\n" + \
             "      model.summary()\n" + \
             "      return model\n\n"
        code.append(py)

#hidden math / ML setup
def training():
    py=" "
    if algorithm == "PG":
        py = "  def remember(self, state, action, prob, reward):\n" + \
             "      y = np.zeros([self.action_size])\n" + \
             "      y[action] = 1\n" + \
             "      self.gradients.append(np.array(y).astype('float32') - prob)\n" + \
             "      self.states.append(state)\n" + \
             "      self.rewards.append(reward)\n\n" + \
             "  def act(self, state):\n" + \
             "      state = state.reshape([1, state.shape[0]])\n" + \
             "      aprob = self.model.predict(state, batch_size=1).flatten()\n" + \
             "      self.probs.append(aprob)\n" + \
             "      prob = aprob / np.sum(aprob)\n" + \
             "      action = np.random.choice(self.action_size, 1, p=prob)\n" + \
             "      return action, prob\n\n" + \
             "  def discount_rewards(self, rewards):\n" + \
             "      discounted_rewards = np.zeros_like(rewards)\n" + \
             "      running_add = 0\n" + \
             "      for t in reversed(range(0, rewards.size)):\n" + \
             "          if rewards[t] != 0:\n" + \
             "              running_add = 0\n" + \
             "          running_add = running_add * self.gamma + rewards[t]\n\n" + \
             "          discounted_rewards[t] = running_add\n" + \
             "      return discounted_rewards\n\n" + \
             "  def train(self):\n" + \
             "      gradients = np.vstack(self.gradients)\n" + \
             "      rewards = np.vstack(self.rewards)\n" + \
             "      rewards = self.discount_rewards(rewards)\n" + \
             "      rewards = rewards / np.std(rewards - np.mean(rewards))\n" + \
             "      gradients *= rewards\n" + \
             "      X = np.squeeze(np.vstack([self.states]))\n" + \
             "      Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))\n" + \
             "      self.model.train_on_batch(X, Y)\n" + \
             "      self.states, self.probs, self.gradients, self.rewards = [], [], [], []\n\n" + \
             "  def load(self, name):\n" + \
             "      self.model.load_weights(name)\n\n" + \
             "  def save(self, name):\n" + \
             "      self.model.save_weights(name)\n\n" + \
             "def preprocess(I):\n" + \
             "   I = I[35:195]\n" + \
             "   I = I[::2, ::2, 0]\n" + \
             "   I[I == 144] = 0\n" + \
             "   I[I == 109] = 0\n" + \
             "   I[I != 0] = 1\n\n" + \
             "   return I.astype(np.float).ravel()\n"
        code.append(py)


    if algorithm == 'QL':
        py = "  def optimizer(self):\n" + \
             "      a = K.placeholder(shape=(None,), dtype='int32')\n" + \
             "      y = K.placeholder(shape=(None,), dtype='float32')\n\n" + \
             "      py_x = self.model.output\n" + \
             "      a_one_hot = K.one_hot(a, self.action_size)\n" + \
             "      q_value = K.sum(py_x * a_one_hot, axis=1)\n" + \
             "      error = K.abs(y - q_value)\n" + \
             "      quadratic_part = K.clip(error, 0.0, 1.0)\n" + \
             "      linear_part = error - quadratic_part\n" + \
             "      loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)\n" + \
             "      optimizer = RMSprop(lr =0.00025, epsilon=0.01)\n" + \
             "      updates = optimizer.get_updates(self.model.trainable_weights, [], loss)\n" + \
             "      train = K.function([self.model.input, a, y], [loss], updates=updates)\n\n" + \
             "      return train\n\n" + \
             "  def update_target_model(self):\n" + \
             "      self.target_model.set_weights(self.model.get_weights())\n\n" + \
             "  def get_action(self, history):\n" + \
             "      history = np.float32(history / 255.0)\n" + \
             "      if np.random.rand() <= self.epsilon:\n" + \
             "          return random.randrange(self.action_size)\n" + \
             "      else:\n" + \
             "          q_value = self.model.predict(history)\n" + \
             "          return np.argmax(q_value[0])\n\n" + \
             "  def replay_memory(self, history, action, reward, next_history, dead):\n" + \
             "      self.memory.append((history, action, reward, next_history, dead))\n\n" \
             "  def train_replay(self):\n" + \
             "      if len(self.memory) < self.train_start:\n" + \
             "          return\n" + \
             "      if self.epsilon > self.epsilon_end:\n" + \
             "          self.epsilon -= self.epsilon_decay_step\n" + \
             "      mini_batch = random.sample(self.memory, self.batch_size)\n" + \
             "      history = np.zeros((self.batch_size, self.state_size[0],self.state_size[1], self.state_size[2]))\n" + \
             "      next_history = np.zeros((self.batch_size, self.state_size[0],self.state_size[1], self.state_size[2]))\n" + \
             "      target = np.zeros((self.batch_size,))\n" + \
             "      action, reward, dead = [], [], []\n" + \
             "      for i in range(self.batch_size):\n" + \
             "          history[i] = np.float32(mini_batch[i][0] / 255.)\n" + \
             "          next_history[i] = np.float32(mini_batch[i][3] / 255.)\n" + \
             "          action.append(mini_batch[i][1])\n" + \
             "          reward.append(mini_batch[i][2])\n" + \
             "          dead.append(mini_batch[i][4])\n" + \
             "      target_value = self.target_model.predict(next_history)\n\n" + \
             "      for i in range(self.batch_size):\n" + \
             "          if dead[i]:\n" +  \
             "              target[i] = reward[i]\n" + \
             "          else:\n" + \
             "              target[i] = reward[i] + self.discount_factor * np.amax(target_value[i])\n" + \
             "      loss = self.optimizer([history, action, target])\n" + \
             "      self.avg_loss += loss[0]\n\n" + \
             "  def save_model(self, name):\n"+ \
             "      self.model.save_weights(name)\n\n" + \
             "  def setup_summary(self):\n" + \
             "      episode_total_reward = tf.Variable(0.)\n" + \
             "      episode_avg_max_q = tf.Variable(0.)\n" + \
             "      episode_duration = tf.Variable(0.)\n" + \
             "      episode_avg_loss = tf.Variable(0.)\n" + \
             "      tf.summary.scalar('Total Reward/Episode', episode_total_reward)\n" + \
             "      tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)\n" + \
             "      tf.summary.scalar('Duration/Episode', episode_duration)\n" + \
             "      tf.summary.scalar('Average Loss/Episode', episode_avg_loss)\n" + \
             "      summary_vars = [episode_total_reward, episode_avg_max_q,episode_duration, episode_avg_loss]\n" + \
             "      summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]\n" + \
             "      update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]\n" + \
             "      summary_op = tf.summary.merge_all()\n" + \
             "      return summary_placeholders, update_ops, summary_op\n\n" + \
             "def pre_processing(observe):\n" + \
             "  processed_observe = np.uint8(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)\n" + \
             "  return processed_observe\n\n"
        code.append(py)


def main():
    py=" "
    if algorithm == 'PG':
        py = "if __name__ == '__main__':\n" + \
             "  env = gym.make('Pong-v0')\n" + \
             "  state = env.reset()\n" + \
             "  prev_x = None\n" + \
             "  score = 0\n" + \
             "  episode = 0\n" + \
             "  state_size = 80 * 80\n" + \
             "  action_size = env.action_space.n\n" + \
             "  agent = PGAgent(state_size, action_size)\n" + \
             "  agent.load('./save_model/pong_reinforce.h5')\n" + \
             "  while True:\n" + \
             "      env.render()\n" + \
             "      cur_x = preprocess(state)\n" + \
             "      x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)\n" + \
             "      prev_x = cur_x\n"
        code.append(py)

    if algorithm == 'QL':
        py = 'if __name__ == "__main__":\n' + \
             '  env = gym.make("BreakoutDeterministic-v4")\n' + \
             '  agent = DQNAgent(action_size=3) # 3\n' + \
             '  scores, episodes, global_step = [], [], 0\n' + \
             '  for e in range(EPISODES):\n' + \
             '      done = False\n' + \
             '      dead = False\n' + \
             '      step, score, start_life = 0, 0, 5\n\n' + \
             '  observe = env.reset()\n' + \
             '  for _ in range(random.randint(1, agent.no_op_steps)):\n' + \
             '      observe, _, _, _ = env.step(1)\n\n' + \
             '  state = pre_processing(observe)\n' + \
             '  history = np.stack((state, state, state, state), axis=2)\n' + \
             '  history = np.reshape([history], (1, 84, 84, 4))\n' + \
             '  while not done:\n' + \
             '      env.render()\n' + \
             '      global_step += 1\n' + \
             '      step += 1\n'
        code.append(py)




def find_probPG():

    py = '    action, prob = agent.act(x)\n'
    code.append(py)

def predict_movesPG():
    py = '      state, reward, done, info = env.step(action)\n' + \
         '      score += reward\n' + \
         '      agent.remember(x, action, prob, reward)\n'
    code.append(py)

def fitPG():
    py = '      if done:\n' + \
         '          episode += 1\n' + \
         '          agent.train()\n' + \
         '          print("Episode: %d - Score: %f." % (episode, score))\n' + \
         '          score = 0\n' + \
         '          state = env.reset()\n' + \
         '          prev_x = None\n' + \
         '          if episode > 1 and episode % 10 == 0:\n' + \
         '              agent.save("./save_model/pong_reinforce.h5")\n'
    code.append(py)

def predictmovesQL():

    py =     '      action = agent.get_action(history)\n' + \
             '      if action == 0:\n' + \
             '          real_action = 1\n' + \
             '      elif action == 1:\n' + \
             '          real_action = 2\n' + \
             '      else:\n' + \
             '          real_action = 3\n' + \
             '      observe, reward, done, info = env.step(real_action)\n' + \
             '      next_state = pre_processing(observe)\n' + \
             '      next_state = np.reshape([next_state], (1, 84, 84, 1))\n' + \
             '      next_history = np.append(next_state, history[:, :, :, :3], axis=3)\n' + \
             '      agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])\n' + \
             '      if start_life > info["ale.lives"]:\n' + \
             '          dead = True\n' + \
             '          start_life = info["ale.lives"]\n'
    code.append(py)

def calculateQvalues():
    py =     '      reward = np.clip(reward, -1., 1.)\n' + \
             '      agent.replay_memory(history, action, reward, next_history, dead)\n' + \
             '      agent.train_replay()\n' + \
             '      if global_step % agent.update_target_rate == 0:\n' + \
             '          agent.update_target_model()\n' + \
             '      score += reward\n' + \
             '      if dead:\n' + \
             '          dead = False\n' + \
             '      else:\n' + \
             '          history = next_history \n' + \
             '      if done:\n' + \
             '          if global_step > agent.train_start:\n' + \
             '              stats = [score, agent.avg_q_max / float(step), step,agent.avg_loss / float(step)]\n' + \
             '              for i in range(len(stats)):\n' + \
             '                  agent.sess.run(agent.update_ops[i], feed_dict={ agent.summary_placeholders[i]: float(stats[i])})\n' + \
             '              summary_str = agent.sess.run(agent.summary_op)\n' + \
             '              agent.summary_writer.add_summary(summary_str, e + 1)\n' + \
             '          print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon,"  global_step:", global_step, "  average_q:", agent.avg_q_max / float(step), "  average loss:", agent.avg_loss / float(step))\n' + \
             '          agent.avg_q_max, agent.avg_loss = 0, 0\n' + \
             '  if e % 1000 == 0:\n' + \
             '      agent.model.save_weights("./save_model/breakout_dqn.h5")\n'
    code.append(py)

def generate():
     final_code = code[0]
     for block in code:
         if final_code != block:
             final_code += block
     if algorithm == 'PG':
         file = open("OURpong.py", 'x')
     elif algorithm == 'QL':
         file = open("OURbreakout_dqn.py", 'x')

     file.write(final_code)
     file.close()







