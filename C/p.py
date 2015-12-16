from ctypes import *
from numpy.ctypeslib import ndpointer

cdll.LoadLibrary('WorkingNetwork.so')

libc = CDLL('WorkingNetwork.so')

pyarr = list(range(784))
arr = (c_double * len(pyarr))(*pyarr)

# libc.neuralNetwork.restype = c_double
libc.neuralNetwork.restype = ndpointer(dtype=c_double, shape=(10,))
print(libc.neuralNetwork(arr))