import h5py

f = h5py.File("dataset.hdf5", 'r')

print(f['0/tty_chars'][0])
print(f['0/tty_colors'][0])
print(f['0/tty_cursor'][0])
print(f['0/action'][0])
print(f['0/strategy'][0])