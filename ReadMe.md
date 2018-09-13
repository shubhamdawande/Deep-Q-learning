## README

This is an implentation of Deep Q learning to play OpenAI Atari SpaceInvadors game.<br />
Q learning algorithm is implemented using a Deep Convolutional Network with 3 Convolutional layers(no pooling)
and a fully connected layer<br />

### Implementation Details
Input: We take a stack of 4 frames at each time step as input state<br />
action space: discrete<br />
state space: continuous<br />

### Overview:

1. Preprocess input frame and build a stack of 4
2. Create a memory for storing state,action,reward,next_state tuples
3. Build our CNN. Network output is the estimated Q values(Q values: accumulated future reward) for a state and possible actions
4. Train our model on frame stack at each step 
   (1) Select an action using epsilon greedy strategy
    ==> Choose random action or an action which maximizes the resulting Q values
   (2) Apply the action and update the memory with tuple <state, action, instant reward, next state>
   (3) Train the NW by sampling input from memory and output labels as target Q values(maximum possible Q values for given state). 

### Evaluation
1) Change training = False to test the learnt model
2) model/tensorboard graphs are saved in models/tensorboard folders
3) After training for 100 episodes our model can achieve average score of ~200
