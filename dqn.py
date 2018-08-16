# import dependencies
import numpy as np
import random
import time
import tensorflow as tf
from skimage import color
from skimage.transform import resize
import matplotlib.pyplot as plt
from collections import deque
import warnings
try:
  import gym
except:
  #!pip install gym
  import gym
try:
  import atari_py
except:
  #!pip install cmake
  #!pip install gym[atari]
  import atari_py

warnings.filterwarnings('ignore')

## Initialize environment
env = gym.make('SpaceInvaders-v0')
env.reset()

############ Tuning parameters ############

training = False
num_episodes = 100
num_steps = 50000
batch_size = 64

# large epsilon->exploration, small epsilon->exploitation
epsilon = 1.0
epsilon_max = 1.0 
epsilon_min = 0.01
decay_rate = 0.00001

stack_len = 4
state_size = [110, 84, stack_len] # downsized for lower computation time
action_size = env.action_space.n
learning_rate = 0.00025
gamma = 0.9

memory_size = 1000000
pretrain_len = batch_size
possible_actions = np.identity(action_size) # convert actions to one-hot vectors

############ Frame preprocessing ############

def preprocess_frame(frame):
  frame_gray       = color.rgb2gray(frame)   # convert to grayscale [210,160,1]
  frame_normalized = frame_gray/255.0        # normalize image in 0 to 1
  frame_resized    = resize(frame_normalized, [110,84]) 
  return frame_resized

############ Stack frames in a deque of length 4############

stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_len)], maxlen=4)

def update_stack(frame, stacked_frames, episode_start):
  
  frame = preprocess_frame(frame)

  if episode_start==True:
    stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_len)], maxlen=4)
    stacked_frames.append(frame)
    stacked_frames.append(frame)
    stacked_frames.append(frame)
    stacked_frames.append(frame)
    stack = np.stack(stacked_frames, axis=2)
  else:
    stacked_frames.append(frame) # add frame t
    stack = np.stack(stacked_frames, axis=2)

  return stack, stacked_frames

############  Memory class ############

class Memory():
  def __init__(self, max_size):
    self.buffer = deque(maxlen = max_size)
    
  def add(self, experience):
    self.buffer.append(experience)
    
  def sample(self, batch_size):
    buffer_size = len(self.buffer)
    index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
    return [self.buffer[i] for i in index]

memory = Memory(memory_size)

############ Initialize memory ############

for i in range(pretrain_len):
  if i==0:
    frame = env.reset()
    stack, stacked_frames = update_stack(frame, stacked_frames, True)
  
  action = env.action_space.sample()
  next_frame, reward, done, info = env.step(action)
  next_stack, stacked_frames = update_stack(next_frame, stacked_frames, False)
  action = possible_actions[action]

  if done: # if game is finished
    done = False
    next_stack = np.zeros(stack.shape)
    memory.add((stack, action, reward, next_stack, done*1))
    frame = env.reset()
    stack, stacked_frames = update_stack(frame, stacked_frames, True)
  else:
    memory.add((stack, action, reward, next_stack, done*1))
    stack = next_stack

############ Build our Network ############

class DeepQNetwork:
  
  def __init__(self, state_size, action_size, learning_rate, name='DQN'):
    
    self.state_size  = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    
    with tf.variable_scope(name):
      # Input/Output placeholders
      self.x  = tf.placeholder(tf.float32, [None, state_size[0], state_size[1], state_size[2]], name='input') # state input: batches x h x w x 4
      self.actions = tf.placeholder(tf.float32, [None, self.action_size], name='actions') # action input: batches x action_size
      self.target_Q = tf.placeholder(tf.float32, [None], name='target_Q')
    
      # 3 convolution layers
      self.layer1 = self.conv_layer(self.x, self.state_size[2], 32, [8,8], name='layer1')
      self.layer2 = self.conv_layer(self.layer1, 32, 64, [4,4], name='layer2')
      self.layer3 = self.conv_layer(self.layer2, 64, 64, [3,3], name='layer3')
      
      # Flattening
      self.flattened = tf.contrib.layers.flatten(self.layer3)
      
      self.layer4 = tf.layers.dense(inputs = self.flattened, units=512, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="layer4")
            
      self.y_ = tf.layers.dense(inputs = self.layer4, kernel_initializer=tf.contrib.layers.xavier_initializer(), units=self.action_size, 
                                           activation=None)
    
      self.pred_Q = tf.reduce_sum(tf.multiply(self.y_, self.actions))

      # Loss function
      self.squared_loss = tf.reduce_mean(tf.square(self.target_Q - self.pred_Q)) # squared sum over batches

      # Optimizer
      self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.squared_loss)

  ## Template for convolution+pooling layer
  def conv_layer(self, input_data, num_input_channels, num_filters, conv_filter_shape, name):
  
    conv_filter_size = [conv_filter_shape[0], conv_filter_shape[1], num_input_channels, num_filters] 
  
    # Conv layer
    kernel = tf.get_variable(
             initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
             shape=conv_filter_size,
             name=name+'_kernel')
    conv_output  = tf.nn.conv2d(input_data, kernel, [1,2,2,1], padding='VALID')
    conv_output  = tf.nn.elu(conv_output)
  
    return conv_output

tf.reset_default_graph()
DeepQNetwork = DeepQNetwork(state_size, action_size, learning_rate) # instantiate class

# TensorBoard Setup
writer = tf.summary.FileWriter("./tensorboard/dqn/1")
#writer.add_graph(sess.graph)
tf.summary.scalar("Loss", DeepQNetwork.squared_loss)
write_op = tf.summary.merge_all()  

############ Training ############

saver = tf.train.Saver()
if training == True:

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    decay_step = 0
    reward_list = []
    
    for episode in range(num_episodes):
      
      # Get initial state (->rgb frame)
      frame = env.reset()
      # Add frame to stack
      stack, stacked_frames = update_stack(frame, stacked_frames, True)
      step = 0
      episode_reward = 0
      total_reward = 0

      while step < num_steps:
        
        step += 1
        #print 'step:', step
        ## Choose optimal action using epsilon greedy strategy
        random_number = random.uniform(0,1)
        decay_step += 1
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * decay_step)
        if random_number < epsilon:
          action = env.action_space.sample()
        else:
          k1,k2,k3 = stack.shape
          action = np.argmax(sess.run(DeepQNetwork.y_, feed_dict={DeepQNetwork.x: stack.reshape(1,k1,k2,k3)})) # input size: [batches,state_size]

        ## Perform the action and update stack with next frame
        next_frame, reward, done, info = env.step(action) # next_frame: HxWx3 
        action = possible_actions[action]

        # Rewards
        episode_reward += reward 
        
        # Append to memory
        if done:
          done = False
          next_frame = np.zeros((state_size[0], state_size[1]),  dtype=np.int)
          next_stack, stacked_frames = update_stack(next_frame, stacked_frames, False)
          step = num_steps
          memory.add((stack, action, reward, next_stack, done*1))
          total_reward = episode_reward
          reward_list.append([episode, episode_reward])

          print('Episode: {}'.format(episode),
                'Total reward: {}'.format(total_reward),
                'epsilon: {:.4f}'.format(epsilon),
                'Training Loss {:.4f}'.format(squared_loss))
        else:
          next_stack, stacked_frames = update_stack(next_frame, stacked_frames, False) # stack: HxWx4
          memory.add((stack, action, reward, next_stack, done*1))
          stack = next_stack

        ## Learning the network weights
        mini_batch       = memory.sample(batch_size)
        stack_batch      = np.array([batch[0] for batch in mini_batch], ndmin=3)
        action_batch     = np.array([batch[1] for batch in mini_batch])
        reward_batch     = np.array([batch[2] for batch in mini_batch])
        next_stack_batch = np.array([batch[3] for batch in mini_batch])
        done_batch       = np.array([batch[4] for batch in mini_batch]).reshape(batch_size,1)
        target_Q_batch = []
        
        # Find max Q values from all actions for next state in a mini batch 
        next_Q_batch = sess.run(DeepQNetwork.y_, feed_dict={DeepQNetwork.x: next_stack_batch})

        for j in range(batch_size):
          
          if done_batch[j] == 1:
            target_Q_batch.append(reward_batch[j])
          else:
            target_Q_batch.append(reward_batch[j] + gamma*np.max(next_Q_batch[j]))

        target_Q = np.array([k for k in target_Q_batch])
        squared_loss, _ = sess.run([DeepQNetwork.squared_loss, DeepQNetwork.adam_optimizer], 
                                   feed_dict={DeepQNetwork.x: stack_batch, 
                                              DeepQNetwork.target_Q: target_Q, 
                                              DeepQNetwork.actions: action_batch})        
        
        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={DeepQNetwork.x: stack_batch,
                                                DeepQNetwork.target_Q: target_Q,
                                                DeepQNetwork.actions: action_batch})
        writer.add_summary(summary, episode)
        writer.flush()
        
    # Save model every 5 episodes
      if episode % 5 == 0:
        save_path = saver.save(sess, "./models/model.ckpt")
        print("Model Saved")

with tf.Session() as sess:
  total_test_rewards = []
    
  # Load the model
  saver.restore(sess, "./models/model.ckpt")
    
  for episode in range(100):
    total_rewards = 0
        
    frame = env.reset()
    stack, stacked_frames = update_stack(frame, stacked_frames, True)
        
    print("****************************************************")
    print("EPISODE ", episode)
    step = 0
    while True:

      # Reshape the state
      stack = stack.reshape((1, state_size[0], state_size[1], state_size[2]))
      # Get action from Q-network 
      # Estimate the Qs values state
      Q_value = sess.run(DeepQNetwork.y_, feed_dict = {DeepQNetwork.x: stack})
            
      # Take the biggest Q value (= the best action)
      action = np.argmax(Q_value)
            
      #Perform the action and get the next_state, reward, and done information
      next_frame, reward, done, _ = env.step(action)
      env.render()

      action = possible_actions[action]
      total_rewards += reward

      if done:
        print ("Score", total_rewards)
        total_test_rewards.append(total_rewards)
        break
                
      next_stack, stacked_frames = update_stack(next_frame, stacked_frames, False)
      stack = next_stack
    
  print np.mean(total_test_rewards)        
  env.close()