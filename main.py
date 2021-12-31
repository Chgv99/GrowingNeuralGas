import tensorflow as tf
import pandas as pd
from GrowingNeuralGas import GrowingNeuralGas
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

#def test():
#    tf.random.set_seed(23)
#    X = tf.concat([tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([0.0, 0.0]),
#                   tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 0.0]),
#                   tf.random.normal([500, 2], 0.0, 0.25, dtype=tf.float32, seed=1) + tf.constant([1.0, 1.0])], 0)
#    growingNeuralGas = GrowingNeuralGas()
#    growingNeuralGas.fit(X, 5)
#    pass

###

gng = GrowingNeuralGas()
gng.load("F:/Usuarios/Chris/Documentos/PyCharm Projects/GNG2/save/Clientes_Ventas_por_Mayor-27-12-2021-19-29-40")


now = datetime.now()
date = now.strftime("%d-%m-%Y-%H-%M-%S")
#dataset = "Clientes_Ventas_por_Mayor"
dataset = "vinos_todas-PCA_3-escalado"
data_path = "data/"
file_name = dataset + ".csv"

data = pd.read_csv(data_path + file_name)

#####DATASET FILAS#####
# para el sample cluster dejamos 1000
#data = data.sample(frac=1).head(1000)

#####DATASET COLUMNAS#####
#data = data.drop(data.columns[2], axis=1)

#normalized_data = data.copy()
#for col in range(data.shape[1]):
#    normalized_data[data.columns[col]] = (data[data.columns[col]] - data[data.columns[col]].min()) / (data[data.columns[col]].max() - data[data.columns[col]].min())

#np.save(save_path + dataset + "-normalized.npy", normalized_data)

df = tf.convert_to_tensor(data, dtype=tf.float32)

epsilon_a = .5
epsilon_n = .05 #.05
a_max = 8
eta = 10
epochs = 1
alpha = .4
delta = .1

growingNeuralGas = GrowingNeuralGas(epsilon_a=epsilon_a, epsilon_n=epsilon_n, a_max=a_max, eta=eta, alpha=alpha, delta=delta)
growingNeuralGas.fit(df, epochs, date, dataset +
                     "-eA" + "_" + str(epsilon_a) +
                     "-eN" + "_" + str(epsilon_n) +
                     "-aM" + "_" + str(a_max) +
                     "-eta" + "_" + str(eta) +
                     "-epochs" + "_" + str(epochs) +
                     "-alpha" + "_" + str(alpha) +
                     "-delta" + "_" + str(delta) +
                     "-" + date)

save_path = "save/" + dataset + "-a" + "_" + str(epsilon_a) + "-amax" + "_" + str(a_max) + "-n" + "_" + str(epsilon_n) + "-e" + "_" + str(eta) +  "-" + date
growingNeuralGas.save(save_path, matlab=True)

pabloF = growingNeuralGas.getGraphConnectedComponents()[0]
pabloO = len(growingNeuralGas.groups())

