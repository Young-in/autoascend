import h5py
import numpy as np

f = h5py.File("/storage/youngin/nld_aaa_small.hdf5", 'r')

n = 0
trans = []
scores = []
turns = []

for k in f.keys():
    n += 1
    trans.append(f[f'{k}/tty_chars'].shape[0])
    scores.append(f[f'{k}/score'][()])
    turns.append(f[f'{k}/turns'][()])

trans = np.array(trans)
scores = np.array(scores)
turns = np.array(turns)

print(f"Total Episodes: {n}")
print(f"Total Transitions: {trans.sum()}")
print(f"Mean Episode Score: {scores.mean()}")
print(f"Median Episode Score: {np.median(scores)}")
print(f"Median Episode Transitions: {np.median(trans)}")
print(f"Median Episode Turns: {np.median(turns)}")