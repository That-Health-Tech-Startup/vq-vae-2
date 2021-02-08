import torch as py
import pandas as pd
import sklearn as sk
import json

mnist_data = pd.read_csv('MNISTLatent.csv')
print(mnist_data)