# DrQ-DRAP

This is a PyTorch implememtation of DrQ and DrQ with DRAP from 

[**Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels**](https://arxiv.org/abs/2004.13649) and 

[**Pre-training of Deep RL Agents for Improved Learning under Domain Randomization**](https://www.researchgate.net/publication/351221844_Pre-training_of_Deep_RL_Agents_for_Improved_Learning_under_Domain_Randomization). 

<p align="center">
        <b>DrQ</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>DrQ+DRAP</b>
</p>

<p align="center">
  <img width="24%" src="gif/75000_f.gif">
  <img width="24%" src="gif/50000_f.gif">
</p>

<p align="center">
  <img width="24%" src="gif/75000_w.gif">
  <img width="24%" src="gif/50000_w.gif">
</p>

## Requirements
To install the requirements run:
```
pip install -r requirements.txt
```
## DrQ Instructions
In DrQ folder there are the files to train the agent (*'walker'* or *'finger'*) with a specific task (*'walk'* or *'spin'*). They can be changed in the ```main.py``` file

To start the training run:
```
python main.py 
```
Monitor results:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | E: X | S: X | R: X | D: X s | BR: X | ALOSS: X | CLOSS: X | TLOSS: X | TVAL: X | AENT: X
```
a training entry decodes as
```
train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy
```
while an evaluation entry
```
| eval  | E: X | S: X | R: X
```
contains
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)
```
The outptut are two .csv files with training and eavaluation statistics and the videos of the training and evaluation phases

## DRAP Instructions
In DRAP folder there are the nokebooks file to create the Dataset with Domain Randomized images. It is used to pre-train the encoder. 

## DrQ+DRAP Instructions
After the pre-training the decoder part is discarded, and the pre-trained encoder, is directly fine-tuned during DrQ training. Since the previous training have been done on 300,000 environment steps, the pre-trained encoder’s weights are used to initialize DrQ encoder and use it for the remaining 200,000 environment interactions.
Again, agent and task cam be changed in the ```main.py``` file.

To start the training run:
```
python main.py 
```
Monitor results:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | E: X | S: X | R: X | D: X s | BR: X | ALOSS: X | CLOSS: X | TLOSS: X | TVAL: X | AENT: X
```
a training entry decodes as
```
train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy
```
while an evaluation entry
```
| eval  | E: X | S: X | R: X
```
contains
```
E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)
```
The outptut are two .csv files with training and eavaluation statistics and the videos of the training and evaluation phases

