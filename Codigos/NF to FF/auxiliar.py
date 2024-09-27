import numpy as np

# Crear un array de ejemplo de dimensiones (100, 100)
array = np.random.rand(100, 100)

# Quedarse solo con los primeros 100 valores de la izquierda
valores_izquierda = array[:,0]

# Verificar las dimensiones del nuevo array
print("Dimensiones del nuevo array:", valores_izquierda.shape)