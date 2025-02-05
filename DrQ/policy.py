import os
from tqdm import tqdm
from models import DRQ
import utils
import torch
import utils
from replay_buffer import ReplayBuffer
from logger import Logger
import time

class Policy(object):
    def __init__(self, env, seed=11):
        self.dir = os.getcwd()
        utils.set_seed_everywhere(seed)
        self.device = torch.device('mps')
        self.env= env

        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]

        self.agent = DRQ(obs_shape=self.obs_shape, action_shape=self.action_shape, action_range=self.action_range, device=self.device)

        self.replay_buffer = ReplayBuffer(self.obs_shape, self.action_shape, capacity=100000, device=self.device)

        self.video_recorder = utils.VideoRecorder(self.dir)

        self.logger = Logger(self.dir,True,10000, action_repeat=4)
        
        self.step = 0

    def evaluate(self, eval_episodes=10):
        avg_reward = 0
        for episode in range(eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode==5)) 
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, False)
                #print(action)
                obs, reward, done, _, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step +=1
            
            avg_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        avg_reward = avg_reward/eval_episodes
        #print(f'Average reward of evaluation at step {self.step}: {avg_reward}')
        self.logger.log('eval/episode_reward', avg_reward, self.step)
        self.logger.dump(self.step)
    
    def train(self, train_steps=1000000, eval_step=5000, seed_obs=1000):
        episode = 0
        episode_reward = 0
        episode_step = 1
        done = True
        print("Training started...")
        start_time = time.time()

        while self.step < train_steps:
            
            if done:
                self.logger.log('train/duration',time.time() - start_time, self.step)
                start_time = time.time()
                self.logger.dump(self.step, save=(self.step >= seed_obs))

                #evaluate agent 
                if self.step % eval_step == 0:
                    print("Periodic agent evaluation...")
                    self.evaluate()
                
                #print(f'Episode {episode} ended. Total reward: {episode_reward} at step {self.step}')
                self.logger.log('train/episode_reward', episode_reward, self.step)

                #Reset the environment for a new episode
                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
        
            #sample action for 1000 seed observation with random policy
            if self.step < seed_obs:
                action = self.env.action_space.sample()
                #print(f'sampled action:{action}')
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs)
                    #print(f'actor action:{action}')
            
            #training update
            if self.step >= seed_obs:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _, _ = self.env.step(action)

            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            
            episode_reward += reward

            self.replay_buffer.push(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1              
        
        print(f"Training complete. Videos available.")