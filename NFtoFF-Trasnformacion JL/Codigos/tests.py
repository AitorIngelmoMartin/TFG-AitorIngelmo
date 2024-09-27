

import numpy as np

array_total = np.array([0,0,0])
array1 = np.array([1,2,3])
array2 = np.array([4,5,6])

array_total = np.append(array_total, array1)
array_total = np.append(array_total, array2)


print(array_total)