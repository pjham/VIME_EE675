from vime import vime
from sac import sac_vime
import gym
from env_wrapper import normallized_action_wrapper

if __name__ == '__main__':
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument('-b', '--buffer', type=int, default=1000000, help='Buffer size (default: 1000000).')
    argparser.add_argument('-d', '--discount', type=float, default=0.99, help='Discount factor, gamma (default: 0.99).')
    argparser.add_argument('-e', '--env', type=str, default='Pendulum-v0', help='ID of gym environment (default: Pendulum-v0).')
    argparser.add_argument('-f', '--freq', type=int, default=500, help='An evaluation episode is done at every _freq_ step.')
    argparser.add_argument('-m', '--max', type=int, default=500, help='Max number of steps (default: 500).')
    argparser.add_argument('-v', '--vime', type=int, default=1, help='Number of experiments (default: 20).')
    argparser.add_argument('-p', '--n_epi', type=int, default=10000, help='Number of episodes (default: 10000).')
    argparser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3).')
    arg = argparser.parse_args()

    env = normallized_action_wrapper(gym.make(arg.env))
    if arg.vime==1:
        vime_model = vime(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_size=64,
            min_logvar=-10,
            max_logvar=2,
            learning_rate=1e-4,
            kl_buffer_capacity=10,
            lamda=1e-2,
            update_iteration=arg.freq,
            batch_size=10,
            eta=1e-1
        )
    else:
        vime_model=None
        
    
    test = sac_vime(
        env=env,
        batch_size=100,
        learning_rate=arg.lr,
        exploration=1,
        episode=arg.n_epi,
        gamma=arg.discount,
        alpha=0.2,
        capacity=arg.buffer,
        rho=0.995,
        update_iter=10,
        update_every=50,
        render=False,
        log=False,
        vime_model=vime_model
    )
    test.run()
