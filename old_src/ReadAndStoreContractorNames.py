import numpy as np

with open("contractors.txt", "r") as file:
    names = file.readlines()
    names = [name.strip() for name in names]

print(len(names))

np.save("contractor_names.npy", np.array(names))