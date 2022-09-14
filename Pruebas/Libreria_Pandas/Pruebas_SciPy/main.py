import scipy
import numpy as np
"""
    Hankel orden 2 grado n - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel2.html#scipy.special.hankel2
"""
Hankel_orden_2 = scipy.special.hankel2(5,(2+8j), out=None)
print(Hankel_orden_2)