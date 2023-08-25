import gym
from autoascend.env_wrapper import EnvWrapper

import h5py
import numpy as np
from tqdm import trange

env = EnvWrapper(gym.make('NetHackChallenge-v0', character='mon-hum-neu', no_progress_timeout=1000), interactive=True)

env.main()