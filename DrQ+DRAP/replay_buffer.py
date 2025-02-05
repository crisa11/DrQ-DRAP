import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        
        self.pad = nn.ReplicationPad2d(4)
        self.crop_size = obs_shape[-1]

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def push(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obs[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obs[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _random_crop(self, padded):
        """Random crop on padded tensors"""
        batch_size, channels, height, width = padded.shape
        crop_x = torch.randint(0, height - self.crop_size + 1, (batch_size,))
        crop_y = torch.randint(0, width - self.crop_size + 1, (batch_size,))
        
        cropped = torch.stack([
            padded[i, :, crop_x[i]:crop_x[i] + self.crop_size, crop_y[i]:crop_y[i] + self.crop_size]
            for i in range(batch_size)
        ])
        return cropped

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        obs = self.obs[idxs]
        next_obs = self.next_obs[idxs]
        obs_aug = obs.copy()
        next_obs_aug = next_obs.copy()

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        obs_aug = torch.as_tensor(obs_aug, device=self.device, dtype=torch.float32)
        next_obs_aug = torch.as_tensor(next_obs_aug, device=self.device, dtype=torch.float32)

        #Padding
        obs = self.pad(obs)
        next_obs = self.pad(next_obs)
        obs_aug = self.pad(obs_aug)
        next_obs_aug = self.pad(next_obs_aug)
        
        #Random Crop
        obs = self._random_crop(obs)
        next_obs = self._random_crop(next_obs)
        obs_aug = self._random_crop(obs_aug)
        next_obs_aug = self._random_crop(next_obs_aug)

        actions = torch.as_tensor(self.actions[idxs], device = self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device = self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device = self.device)

        return obs, actions, rewards, next_obs, not_dones_no_max, obs_aug, next_obs_aug 
    