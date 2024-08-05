import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Defining the spambase dataset
spambase = fetch_ucirepo(id=94)

# Splitting the dataset into features and target
X = spambase.data.features
target = spambase.data.targets

# Creating entry and train size constants
entrada = tf.constant(X, dtype=tf.float32)
y = tf.constant(target, dtype=tf.float32)

# Splitting the dataset into training and testing sets
X_treino, X_teste, y_treino, y_teste = train_test_split(entrada.numpy(), y.numpy(), test_size=0.2, stratify=y.numpy(), random_state=4321)

# Normalizing the dataset with StandardScaler
scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

# Converting the dataset to tensors
X_treino = tf.constant(X_treino, dtype=tf.float32)
X_teste = tf.constant(X_teste, dtype=tf.float32)
y_treino = tf.constant(y_treino, dtype=tf.float32)
y_teste = tf.constant(y_teste, dtype=tf.float32)

# Defining the get_weights_biases function
def obter_pesos_vies(tamanho_entrada, tamanho_oculta1, tamanho_oculta2):
  tf.random.set_seed(31)

  pesos1 = tf.Variable(tf.random.normal([tamanho_entrada, tamanho_oculta1]))
  vies1 = tf.Variable(tf.random.normal([tamanho_oculta1]))

  pesos2 = tf.Variable(tf.random.normal([tamanho_oculta1, tamanho_oculta2]))
  vies2 = tf.Variable(tf.random.normal([tamanho_oculta2]))

  pesos_final = tf.Variable(tf.random.normal([tamanho_oculta2, 1]))
  vies_final = tf.Variable(tf.random.normal([1]))

  return pesos1, vies1, pesos2, vies2, pesos_final, vies_final

# Defining neuron function
def neuronio(x, pesos, vies):
  return tf.add(tf.matmul(x, pesos), vies)

# Hidden layer 1 & 2
neuronios_entrada = X_treino.shape[1]
variaveis = obter_pesos_vies(neuronios_entrada, 6, 4)
pesos1, vies1, pesos2, vies2, pesos_final, vies_final = variaveis

# Loss calculation
calculadora_perda = tf.keras.losses.BinaryCrossentropy()

# Training optimizer
otimizador_treino = tf.optimizers.SGD(learning_rate=0.01)

# Loss and accuracy lists
perdas = []
taxas_acerto = []

# Training loop
for epoca in range(1000): # Epochs = 1000
  with tf.GradientTape() as tape:
    # Using ReLU activation function
    treino1 = tf.nn.relu(neuronio(X_treino, pesos1, vies1))
    treino2 = tf.nn.relu(neuronio(treino1, pesos2, vies2))
    treino3 = tf.sigmoid(neuronio(treino2, pesos_final, vies_final))
    custo = calculadora_perda(y_treino, treino3)

  perdas.append(custo.numpy())

  teste1 = tf.nn.relu(neuronio(X_teste, pesos1, vies1))
  teste2 = tf.nn.relu(neuronio(teste1, pesos2, vies2))
  teste3 = tf.sigmoid(neuronio(teste2, pesos_final, vies_final))

  acertos = np.mean(y_teste.numpy() == ((teste3.numpy() > 0.5) *1))
  taxas_acerto.append(acertos)

  gradientes = tape.gradient(custo, variaveis)
  otimizador_treino.apply_gradients(zip(gradientes, variaveis))

# Plotting the loss and accuracy
plt.plot(perdas)
plt.plot(taxas_acerto)
plt.title('Perdas e taxas de acerto por época com ReLU')
plt.legend(['Perda no treino', 'Taxa de acerto no teste'])
plt.xlabel('Época')
plt.ylabel('Taxa de acerto')
plt.ylim(0,1)
plt.show()