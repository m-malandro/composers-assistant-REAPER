import os
import pickle

with open('zzzzz_failed.txt', 'rb') as infile:
    F = pickle.load(infile)

for T in F:
    to_kill = T[0]
    # print(to_kill)
    # print(T[1])
    # print()
    os.remove(to_kill)

