import numpy as np
from io import StringIO

data_file = open("velLandscape.txt", "r")
data_str = data_file.read()

# Convert to numpy array
data = np.loadtxt(StringIO(data_str), delimiter=",")

print(data.shape)
print(data[:])
