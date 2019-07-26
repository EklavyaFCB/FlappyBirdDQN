#!/usr/bin/env python

import sys
import cv2
import random
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import deque
sys.path.append("game/")
import flappy_bird as game
import matplotlib.pyplot as plt

# Variables
NUM_ACTIONS = 2 # Total number of actions

# Hyperparameters
GAMMA = 0.99
NUM_OBSERVATIONS = 10000 # Number of timesteps to observe before starting training
NUM_EXPLORATIONS = 300000 # Frames over which to anneal epsilon
INITIAL_EPSILON = 0.01
FINAL_EPSILON = 0.00001 
REPLAY_MEMORY_SIZE = 50000 # Number of experiences to store
BATCH = 32
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6

# We want the same datetime variable for both
myDateTime = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())

# Methods
def setup_logger(logger_name, log_file, log_format='%(message)s', level=logging.INFO):
    
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(log_format)
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)

def make_two_loggers():

    ''' 
    Makes two loggers, config_log and my_log, and returns the latter.
    The former is used in the method log_config() to log the (hyper-)paremeters.
    '''

    # First logger
    config_log_name = 'logs/config/config_{}.log'.format(myDateTime)
    setup_logger('config_log', config_log_name)
    config_log = logging.getLogger('config_log')

    # Make config .log file
    log_config(config_log)

    # Second logger
    my_log_name = 'logs/stdout/log_{}.log'.format(myDateTime)
    setup_logger('my_log', my_log_name)
    my_log = logging.getLogger('my_log')

    return my_log

def log_config(my_log):
    ''' Make a log file of the (hyper-)parameters'''
    my_log.info('GAMMA: '+ str(GAMMA))
    my_log.info('NUM_OBSERVATIONS: '+ str(NUM_OBSERVATIONS))
    my_log.info('NUM_EXPLORATIONS: '+ str(NUM_EXPLORATIONS))
    my_log.info('INITIAL_EPSILON: '+ str(INITIAL_EPSILON))
    my_log.info('FINAL_EPSILON: '+ str(FINAL_EPSILON))
    my_log.info('LEARNING_RATE: '+ str(LEARNING_RATE))
    my_log.info('REPLAY_MEMORY_SIZE: '+ str(REPLAY_MEMORY_SIZE))
    my_log.info('BATCH: '+ str(BATCH))
    my_log.info('FRAME_PER_ACTION: '+ str(FRAME_PER_ACTION))

def make_tf_weight_var(shape):
    '''
    Returns a tensorflow Variable of a tensor of random samples from a normal distribution, in the given shave, which are used as weights for the CNN-model.
    '''
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def make_tf_bias_var(shape):
    '''Returns a tensorflow variable of a bias constant in the given shape.'''
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    '''Computes and returns a 2D convolution given 4D input and filter tensors'''
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    '''Performs and returns the max pooling on the input'''
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def determine_state(t):
    '''
    Determines and returns which state the model is currently in, 
    by comparing the timestamp t to the global variables NUM_OBSERVATIONS 
    and NUM_EXPLORATIONS.
    '''
    if t <= NUM_OBSERVATIONS:
        state = 'Observation'
    elif t > NUM_OBSERVATIONS and t <= NUM_OBSERVATIONS + NUM_EXPLORATIONS:
        state = 'Exploration'
    else:
        state = 'Training'

    return state

def create_model():
    '''Implements and returns a CNN network of the, with ReLU as the non-linear activation for hidden layers.'''

    # Input layer - a (80,80,historyLength) image and single output for every possible action
    s = tf.placeholder("float", [None, 80, 80, 4])
    
    # First convolution layer - 32 filters, size (8,8), stride 4, followed by ReLU
    W_conv1 = make_tf_weight_var([8, 8, 4, 32])
    b_conv1 = make_tf_bias_var([32])
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Second convolution layer - 64 filters, size (4,4), stride 2, followed by ReLU
    W_conv2 = make_tf_weight_var([4, 4, 32, 64])
    b_conv2 = make_tf_bias_var([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer - 64 filters, size (3,3), stride 1, followed by ReLU
    W_conv3 = make_tf_weight_var([3, 3, 64, 64])
    b_conv3 = make_tf_bias_var([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)
    
    # Following that is a fully connected layer with 512 outputs
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    W_fc1 = make_tf_weight_var([1600, 512])
    b_fc1 = make_tf_bias_var([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    
    # The output layer (also fully connected) with a single output for each action.
    W_fc2 = make_tf_weight_var([512, NUM_ACTIONS])
    b_fc2 = make_tf_bias_var([NUM_ACTIONS])

    # Output layer - the values here represent the Q-function given the input state for each valid action. At each time step, the model performs whichever action corresponds to the highest Q-value using a epsilon-greedy policy.
    output = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, output, h_fc1

def train_network(s, output, h_fc1, sess, my_log):
    
    # Loss function variables
    tf_a = tf.placeholder("float", [None, NUM_ACTIONS])
    tf_y = tf.placeholder("float", [None])
    output_action = tf.reduce_sum(tf.multiply(output, tf_a), reduction_indices=1)

    # Cost variable which we will attempt to optimise
    tf_cost = tf.reduce_mean(tf.square(tf_y - output_action))

    # We will AdamOptimizer to optimse our 'cost' variable,
    # by calling train_step.run() later with variables y, a, s
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(tf_cost)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=1e-6, decay=0.9, momentum=0.95).minimize(tf_cost)

    # Open up a game state to communicate with emulator
    game_state = game.GameState()

    # Initialise replay memory to store previous observations
    replayMemory = deque()

    # Get the first state by doing nothing
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[0] = 1
    
    # Preprocess initial image to 80x80x4
    # frame_step() returns image_data (x_t), reward (r_0), terminal
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("checkpoints")
    
    # RESTORE OLD CHECKPOINT
    if checkpoint and checkpoint.model_checkpoint_path:
        #saver.restore(sess, checkpoint.model_checkpoint_path)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/')) # Load latest checkpoint
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights.")

    # Start training
    epsilon = INITIAL_EPSILON
    tot_reward = 0
    best_reward = 0
    t = 0

    dataString = 'logs/rewards/train/ep_rewards_{}.txt'.format(myDateTime)
    myfile = open(dataString, 'w')

    dataString = 'logs/rewards/test/ep_high_scores_{}.txt'.format(myDateTime)
    myfile2 = open(dataString, 'w')
    
    while True:
        
        # Choose an action with epsilon-greedy policy
        output_t = output.eval(feed_dict={s : [s_t]})[0] # e.val() is a shortcut for tf.get_default_session().run(t)
        a_t = np.zeros([NUM_ACTIONS])
        action_index = 0
        
        if t % FRAME_PER_ACTION == 0:
            
            # Roll dice
            # If rand value is lower than epsilon, then choose action randomly
            if np.random.random() <= epsilon:
                action_index = np.random.randint(NUM_ACTIONS)
                a_t[np.random.randint(NUM_ACTIONS)] = 1
            
            # Else choose action with highest value
            else:
                action_index = np.argmax(output_t)
                a_t[action_index] = 1
        
        else:
            a_t[0] = 1 # Do nothing

        # Decrement epsilon (if we have finished observing)
        if epsilon > FINAL_EPSILON and t > NUM_OBSERVATIONS:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EXPLORATIONS

        # Run the selected action, and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        
        # Process next image to greyscale, and rescale to 80x80x4
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # Add experience to replay memory replayMemory
        # e = (s_t, a_t, r_t, s_t1), but we also add the 'terminal' variable
        replayMemory.append((s_t, a_t, r_t, s_t1, terminal))
        
        # If the length of replay memory becomes bigger than our chosen limit, pop left sided values
        if len(replayMemory) > REPLAY_MEMORY_SIZE:
            replayMemory.popleft()

        # Once the model has observed enough states, we can start exploring/training
        if t > NUM_OBSERVATIONS:
            
            # Sample a minibatch from the replay to train on
            minibatch = random.sample(replayMemory, BATCH)

            # Get the batch variables by columns - state, action, reward, next state
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            
            # e.val() is a shortcut for tf.get_default_session().run(t)
            output_j1_batch = output.eval(feed_dict = {s : s_j1_batch})
            
            # Iterate through minibatch, and update DQN
            for i in range(len(minibatch)):
                
                # Get terminal value from minibatch
                terminal = minibatch[i][-1]
                
                # If terminal, y only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                
                # If not terminal, y equals reward + gamma * max(output_j1_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(output_j1_batch[i]))

            # Perform gradient step
            train_step.run(feed_dict = {
                tf_y : y_batch,
                tf_a : a_batch,
                s : s_j_batch}
            )

        # Update state and timestamp
        t += 1
        s_t = s_t1
        tot_reward += r_t

        # Update best high score if necessary
        if tot_reward > best_reward:
            best_reward = tot_reward

        # Save progress every 10k iterations
        if t % (NUM_OBSERVATIONS/10) == 0 or t==1:
            saver.save(sess, 'checkpoints/step', global_step = t)

            # Determine state
            state = determine_state(t)

            # Save progress to log at every iteration
            my_log.info('TIMESTEP ' + str(t) + '/ STATE ' + str(state) + '/ EPSILON ' + str(np.round(epsilon, 2)) + '/ ACTION ' + str(action_index) + '/ REWARD ' + str(r_t) + '/ TOT REWARD ' + str(np.round(tot_reward,2)) + '/ BEST REWARD ' + str(np.round(best_reward,2)) + '/ Q_MAX %s ' % str(np.max(output_t)))
            
            # Print info
            print('TIMESTEP', t, '/ STATE', state, '/ EPSILON', np.round(epsilon, 2), '/ ACTION', action_index, '/ REWARD', r_t, '/ TOT REWARD', np.round(tot_reward,2), '/ BEST REWARD', np.round(best_reward,2), '/ Q_MAX %e' % np.max(output_t))

        # Change total reward to zero
        if terminal:
            myfile.write("%.1f\n" % tot_reward)
            myfile2.write("%.1f\n" % best_reward)
            tot_reward = 0

            #if t>NUM_EXPLORATIONS:
            #    myfile.close()
            #    break

def test(s, output, h_fc1, sess):

    # Open up a game state to communicate with emulator
    game_state = game.GameState()

    # Get the first state by doing nothing
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[0] = 1
    
    # Preprocess initial image to 80x80x4
    # frame_step() returns image_data (x_t), reward (r_0), terminal
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # Saving and loading networks
    saver = tf.train.Saver()
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("checkpoints")
    
    # RESTORE OLD CHECKPOINT
    if checkpoint and checkpoint.model_checkpoint_path:
        #saver.restore(sess, checkpoint.model_checkpoint_path)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/')) # Load latest checkpoint
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights.")

    # Start training
    epsilon = INITIAL_EPSILON
    tot_reward = 0
    best_reward = 0
    t = 0

    dataString = 'logs/rewards/test/ep_rewards_{}.txt'.format(myDateTime)
    myfile = open(dataString, 'w')

    dataString = 'logs/rewards/test/ep_high_scores_{}.txt'.format(myDateTime)
    myfile2 = open(dataString, 'w')
    
    while True:
        
        # Choose an action with epsilon-greedy policy
        output_t = output.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([NUM_ACTIONS])
        action_index = 0
        
        if t % FRAME_PER_ACTION == 0:
            action_index = np.argmax(output_t)
            a_t[action_index] = 1
        
        else:
            a_t[0] = 1 # Do nothing

        # Run the selected action, and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        
        # Process next image to greyscale, and rescale to 80x80x4
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # Update state and timestamp
        t += 1
        s_t = s_t1
        tot_reward += r_t

        # Update best high score if necessary
        if tot_reward > best_reward:
            best_reward = tot_reward

        # Change total reward to zero
        if terminal:
            myfile.write("%.1f\n" % tot_reward)
            myfile2.write("%.1f\n" % best_reward)
            tot_reward = 0

def playGame():
    my_log = make_two_loggers()
    sess = tf.InteractiveSession()
    s, output, h_fc1 = create_model()
    #train_network(s, output, h_fc1, sess, my_log) # Uncomment to train
    test(s, output, h_fc1, sess) # Comment to train

def main():
    playGame()

if __name__ == "__main__":
    main()




