import gym
from autoascend.env_wrapper import EnvWrapper

import h5py
import numpy as np
from tqdm import trange

env = EnvWrapper(gym.make('NetHackChallenge-v0', character='mon-hum-neu', no_progress_timeout=1000))

with h5py.File('/storage/youngin/nld_aaa.hdf5', 'w') as f:
    n = 100
    n_failed = 0
    for i in trange(n):
        try:
            env.main()
        except Exception as e:
            n_failed += 1
        else:
            history = env.history

            grp = f.create_group(f"{i}")

            epi_len = len(history)

            grp.create_dataset("tty_chars", data = np.concatenate([np.expand_dims(t['tty_chars'], 0) for t in history], axis = 0))
            grp.create_dataset("tty_colors", data = np.concatenate([np.expand_dims(t['tty_colors'], 0) for t in history], axis = 0))
            grp.create_dataset("tty_cursor", data = np.concatenate([np.expand_dims(t['tty_cursor'], 0) for t in history], axis = 0))
            grp.create_dataset("action", data = [t['act'] for t in history])
            grp.create_dataset("strategy", data = [t['strategy'] for t in history])

    print(f"fail ratio:{n_failed / n}")