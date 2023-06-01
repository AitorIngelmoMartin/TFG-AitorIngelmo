import numpy as np

array = np.random.randint(1,50,size=(5,5))
print(array)

posiciones_mayores_a_25 = np.where(array > 25)
array[posiciones_mayores_a_25] = 0
print(array)

prueba = np.zeros((10,10))
print(prueba)