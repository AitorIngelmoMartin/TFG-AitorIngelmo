import numpy as np
import numpy.ma as ma


delta_kx=5
dimension = 5
kxcoordinatesMatrix=np.ones((dimension,dimension))
for j in range(dimension):  
    for i in range(dimension):            
        if i<=(dimension/2):
            indice=i-1
        else:
            indice=i-1-dimension
        kxcoordinatesMatrix[i-1][j-1]=(delta_kx*indice)
print(kxcoordinatesMatrix)

kycoordinatesMatrix = np.rot90(kxcoordinatesMatrix,3)
print("\n",kycoordinatesMatrix)


rad = np.array([ np.sqrt(self.rad[k]) if self.rad[k]>=0 else -np.sqrt(-self.rad[k])*1j for k in range(dimension)])
