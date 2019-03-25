import os
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
#from BrainDQN import *
from utils import MyDequeSeq, CreatePics
from DQN_Model import DeepQNN
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

GAMMA = 0.99
INITIAL_EPSILON = 0.3
FINAL_EPSILON = 0.0001
BATCH_SIZE = 32
TURNS = 100

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

train_para = {
    'OBSERVE' : 10000,
    'EPISODE' : 3000000
}
test_para = {
    'OBSERVE' : 500,
    'EPISODE' : 3
}

replay_mem_file = './cache/replay_memory.pkl'
model_checkpoint_file = './cache/model_checkpoint.pth'
best_model_file = './cache/best_model_checkpoint_%d_steps_%d_turns.pth'

def train_dqn(model=None, filename=model_checkpoint_file, mode='train'):
    '''
    Here mode 'train' means running the whole steps to train our model,
    mode 'test' means testing whether our code can run properly. 
    Please separate 'mode' from the mode of models.
    '''
    if mode == 'train':
        paras = train_para
    else:
        paras = test_para

    OBSERVE = paras['OBSERVE']
    EPISODE = paras['EPISODE']
    epsilon = INITIAL_EPSILON
    episode = 0
    best_steps = 51
    current_step = 0 # times of running
    current_state = None # 4 frames
    # replay memory pool
    replay_memory = MyDequeSeq(filename=replay_mem_file)

    if os.path.exists(filename):
        print('This time you start with some pretrained parameters.')
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        current_step = checkpoint['current_step']
        epsilon = checkpoint['epsilon']
        episode = checkpoint['episode']
        best_steps = checkpoint['best_steps']
        replay_memory.load_mode()
            

    if use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    ceriterion = nn.MSELoss()
    # strat the game
    flappybird = game.GameState()
    replay_memory.set_init_state()
    img_load = CreatePics()

    donothing = [1,0]
    o,r,t = flappybird.frame_step(donothing)
    init_o = img_load.generate_next_input(o)

    next_o = init_o

    if current_step >= OBSERVE:
        print('We will skip the observation procedure.')
    else:
        print('Start obtaining Experience Pool.')
    # observe steps
    while current_step < OBSERVE:
        old_o = next_o
        #print('old_o shape',old_o.shape)
        action = model.get_random_action()
        o, reward, terminal = flappybird.frame_step(action)
        next_o = img_load.generate_next_input(o)

        replay_memory.store_transtition((old_o, action, reward, next_o, terminal))

        if current_step % 5000 == 0:
            replay_memory.save_cache()

        if current_step % 1000 == 0:
            print('Observation Step: %d'%current_step)

        current_step += 1


    print('Observation step Done!')
    print('Start Training!')

    # training
    while episode < EPISODE:
        epsilon = INITIAL_EPSILON
        next_o = init_o
        current_turns = 0
        while True:
            current_o = next_o
            action = model.get_action(epsilon, current_o)
            epsilon = epsilon - (INITIAL_EPSILON - FINAL_EPSILON)/current_step if epsilon > FINAL_EPSILON else FINAL_EPSILON
            o, reward, terminal = flappybird.frame_step(action)
            next_o = img_load.generate_next_input(o)

            replay_memory.store_transtition((current_o, action, reward, next_o, terminal))

            minibatch = random.sample(replay_memory.replay_mem, BATCH_SIZE)

            S0s = np.array([data[0] for data in minibatch])
            Ats = np.array([data[1] for data in minibatch])
            Rws = np.array([data[2] for data in minibatch])
            S1s = np.array([data[3] for data in minibatch])
            Tms = np.array([data[4] for data in minibatch])

            q0_mat = model.forward(S0s)
            q1_mat = model.forward(S1s)

            y = Rws.astype(np.float32)

            for j in range(BATCH_SIZE):
                if not Tms[j]:
                    y[j] += GAMMA * q1_mat[j].max()

            y = Variable(torch.from_numpy(y))
            action_batch = Variable(torch.from_numpy(Ats))

            if use_cuda:
                y = y.cuda()
                action_batch = action_batch.cuda()

            q_value = torch.sum(torch.mul(q0_mat.type(torch.DoubleTensor), action_batch.type(torch.DoubleTensor)), dim=1)

            loss = ceriterion(q_value.type(torch.DoubleTensor), y.type(torch.DoubleTensor))
            loss.backward()

            optimizer.step()

            if current_step % 10000 == 0:
                replay_memory.save_cache()



            current_step += 1

            if terminal:
                break
            current_turns += 1

        episode += 1

        print('This is Episode {}, step {}. Current turns {}.'.format(episode, current_step, current_turns))

        save_checks = {
            'state_dict' : model.state_dict(),
            'current_step' : current_step,
            'best_steps' : best_steps,
            'episode' : episode,
            'epsilon' : epsilon
        }

        if current_turns > best_steps:
            best_steps = current_turns
            torch.save(save_checks, best_model_file%(current_step, best_steps))

        torch.save(save_checks, filename)


def play_game(best_model):
    flappybird = game.GameState()

    model = DeepQNN()
    check = torch.load(best_model)
    model.load_state_dict(check['state_dict'])

    donothing = [1,0]
    o,r,t = flappybird.frame_step(donothing)
    img_load = CreatePics()
    init_o = img_load.generate_next_input(o)

    while True:
        action = model.get_optim_action(init_o)
        o,r,t = flappybird.frame_step(action)
        init_o = img_load.generate_next_input(o)

if __name__ == '__main__':
    dqn = DeepQNN()
    train_dqn(dqn,mode='train')