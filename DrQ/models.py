import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, n_features, device='mps'):
        super().__init__()
        self.device = device
        self.n_features = n_features
        self.img_channels = obs_shape[0]
        self.n_filters = 32

        self.conv1 = nn.Conv2d(self.img_channels, self.n_filters, 3, stride=2)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters, 3, stride=1)
        self.conv3 = nn.Conv2d(self.n_filters, self.n_filters, 3, stride=1)
        self.conv4 = nn.Conv2d(self.n_filters, self.n_filters, 3, stride=1)
        
        self.fc = nn.Linear(35 * 35 * self.n_filters, self.n_features)
        self.norm = nn.LayerNorm(self.n_features)

    def forward(self, obs, detach=False):
        obs = obs / 255.0  
        self.conv1_output = F.relu(self.conv1(obs))
        self.conv2_output = F.relu(self.conv2(self.conv1_output))
        self.conv3_output = F.relu(self.conv3(self.conv2_output))
        self.conv4_output = F.relu(self.conv4(self.conv3_output))
        
        x = self.conv4_output.reshape(self.conv4_output.size(0), -1)

        if detach:
            x = x.detach()

        self.fc_output = self.fc(x)
        self.norm_output = self.norm(self.fc_output)

        out = torch.tanh(self.norm_output)
        return out

    def copy_conv_weights_from(self, source):
        utils.tie_weights(src=source.conv1, trg=self.conv1)
        utils.tie_weights(src=source.conv2, trg=self.conv2)
        utils.tie_weights(src=source.conv3, trg=self.conv3)
        utils.tie_weights(src=source.conv4, trg=self.conv4)

    def log(self, logger, step):
        """
        Log fot the parameters and statistics of the Encoder.

        Args:
            logger (Logger): logger to register data.
            step (int): Current step.
        """
        #Log of outputs
        outputs = {
            'conv1_out': self.conv1_output,
            'conv2_out': self.conv2_output,
            'conv3_out': self.conv3_output,
            'conv4_out': self.conv4_output,
            'fc_out': self.fc_output,
            'norm_out': self.norm_output
        }

        for k, v in outputs.items():
            logger.log_histogram(f'train_encoder/{k}_hist', v, step)
            if len(v.shape) > 2:
                logger.log_image(f'train_encoder/{k}_img', v[0], step)

        #Log of convolutional parameters
        for i, conv_layer in enumerate([self.conv1, self.conv2, self.conv3, self.conv4]):
            logger.log_param(f'train_encoder/conv{i + 1}', conv_layer, step)

        #Log dei parametri del fully connected e LayerNorm
        logger.log_param('train_encoder/fc', self.fc, step)
        logger.log_param('train_encoder/norm', self.norm, step)

class Actor(nn.Module):
    def __init__(self, n_features, action_shape, hidden_size=1024, log_std_min=-10, log_std_max=2, device='mps'):
        super(Actor, self).__init__()

        self.device = device
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(n_features, hidden_size)
        nn.init.orthogonal_(self.linear1.weight.data)
        self.linear1.bias.data.fill_(0.0)
        
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        nn.init.orthogonal_(self.linear2.weight.data)
        self.linear2.bias.data.fill_(0.0)

        self.mean_linear = nn.Linear(hidden_size, action_shape[0])
        nn.init.orthogonal_(self.mean_linear.weight.data)
        self.mean_linear.bias.data.fill_(0.0)

        self.log_std_linear = nn.Linear(hidden_size, action_shape[0])
        nn.init.orthogonal_(self.log_std_linear.weight.data)
        self.log_std_linear.bias.data.fill_(0.0)

    def forward(self, obs, detach_encoder=False):
        #obs = self.encoder(obs, detach=detach_encoder)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)                                                                    
        std = log_std.exp()
        
        dist = utils.SquashedNormal(mu, std)

        return dist  

    def log(self, logger, step):
        """
        Log fot the parameters and statistics of the Actor.

        Args:
            logger (Logger): logger to register data.
            step (int): Current step.
        """
        #Log of feedforward parameters
        logger.log_param('train_actor/linear1', self.linear1, step)
        logger.log_param('train_actor/linear2', self.linear2, step)
        logger.log_param('train_actor/mean_linear', self.mean_linear, step)
        logger.log_param('train_actor/log_std_linear', self.log_std_linear, step)

        #Log of dtributions
        logger.log_histogram('train_actor/mean_weights', self.mean_linear.weight.data, step)
        logger.log_histogram('train_actor/mean_bias', self.mean_linear.bias.data, step)
        logger.log_histogram('train_actor/log_std_weights', self.log_std_linear.weight.data, step)
        logger.log_histogram('train_actor/log_std_bias', self.log_std_linear.bias.data, step) 

class Critic(nn.Module):
    def __init__(self, n_features, action_shape, hidden_size=1024, device='mps'):
        super(Critic, self).__init__()

        self.device = device

        self.Q1 = nn.Sequential (
                nn.Linear(n_features + action_shape[0], hidden_size), 
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 1) #fc3
            ) 

        self.Q2 = nn.Sequential(
            nn.Linear(n_features + action_shape[0], hidden_size), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1) #fc3
        )
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, obs, action, detach_encoder=False):
        #obs = self.encoder(obs, detach=detach_encoder)
        x = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

    def log(self, logger, step):
        """
        Log fot the parameters and statistics of the Critic.

        Args:
            logger (Logger): logger to register data.
            step (int): Current step.
        """
        #Log of  Q1
        logger.log_param('train_critic/Q1_fc1', self.Q1[0], step)
        logger.log_param('train_critic/Q1_fc2', self.Q1[2], step)
        logger.log_param('train_critic/Q1_fc3', self.Q1[4], step)

        #Log of Q2
        logger.log_param('train_critic/Q2_fc1', self.Q2[0], step)
        logger.log_param('train_critic/Q2_fc2', self.Q2[2], step)
        logger.log_param('train_critic/Q2_fc3', self.Q2[4], step)

        # Log of Q1 and Q2
        for q_name, q_values in {'q1': self.Q1[4].weight.data, 'q2': self.Q2[4].weight.data}.items():
            logger.log_histogram(f'train_critic/{q_name}_hist', q_values, step)

class DRQ(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, lr=1e-3, n_features=50, hidden_size= 1024, gamma=0.99, init_alpha=0.1, critic_tau=0.01, actor_update_freq=2, critic_target_update_freq=2, batch_size=128, device=torch.device('mps')):
        
        print("Initializing DRQ...")
        print("Initializing parameters and models...")

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_range = action_range
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.batch_size = batch_size
        self.device = device

        self.encoder = Encoder(self.obs_shape, self.n_features).to(self.device)

        self.actor = Actor(self.n_features, self.action_shape, self.hidden_size).to(self.device)

        self.critic = Critic(self.n_features, self.action_shape, self.hidden_size).to(self.device)
        self.critic_target = Critic(self.n_features, self.action_shape, self.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32).to(device)
        self.log_alpha.requires_grad = True
        
        self.target_entropy = -action_shape[0]

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        obs = self.encoder(obs)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        #print(a, a.shape())
        assert action.ndim == 2 and action.shape[0] == 1
        action  = action.cpu().detach().numpy()
        #print(action[0])
        return action[0]

    def update_critic(self, obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done, logger, step):
        with torch.no_grad():

            #Not augmented
            next_obs = self.encoder(next_obs)
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_prob
            
            target_Q = reward + (self.gamma * target_V * not_done)

            #Augmented
            next_obs_aug = self.encoder(next_obs_aug)
            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug)
            target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_prob_aug
            
            target_Q_aug = reward + (self.gamma * target_V * not_done)

            #Mean 
            final_target_Q = (target_Q + target_Q_aug) / 2
        
        #Current Q estimation and loss
        obs = self.encoder(obs)
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss_not_aug = F.mse_loss(current_Q1, final_target_Q) + F.mse_loss(current_Q2, final_target_Q)

        #Add augmented term to the loss
        obs_aug = self.encoder(obs_aug)
        current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
        critic_loss_aug = F.mse_loss(current_Q1_aug, final_target_Q) + F.mse_loss(current_Q2_aug, final_target_Q)

        critic_loss = critic_loss_not_aug + critic_loss_aug
        
        logger.log('train_critic/loss', critic_loss, step)
        #print(f'Critic loss: {critic_loss}')

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_opt.step()

        self.critic.log(logger, step)

    def update_actor(self, obs, logger, step):

        obs = self.encoder(obs, detach=True) 

        dist = self.actor(obs)
        action = dist.rsample() 
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)
        #print(f'Actor loss: {actor_loss}')
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        #Update alpha
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        #print(f'Alpha loss: {actor_loss}')

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        
    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)
        self.update_critic(obs,obs_aug,action,reward,next_obs,next_obs_aug,not_done, logger, step)

        if step % self.actor_update_freq == 0:
            #print(f'Updating actor at step {step}...')
            self.update_actor(obs, logger, step)
        
        if step % self.critic_target_update_freq == 0:
            #print(f'Updating target critic at step {step}...')
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        