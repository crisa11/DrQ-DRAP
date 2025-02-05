import warnings
import dmc2gym
from policy import Policy
import utils

def create_env(domain, task, height=84 ,width=84, action_repeat=4, frame_stack=3, seed=1):

    camera_id = 0
    
    env = dmc2gym.make(domain_name=domain,
                    task_name=task,
                    seed=seed,
                    visualize_reward=False,
                    from_pixels=True,
                    height=height,
                    width=width,
                    frame_skip=action_repeat,
                    camera_id=camera_id)

    env=utils.FrameStack(env, k=frame_stack)
    
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    #domain = 'finger'
    #task = 'spin'
    domain = 'walker'
    task = 'walk'
    env=create_env(domain, task)
    policy = Policy(env)

    policy.train()

if __name__ == '__main__':
    main()