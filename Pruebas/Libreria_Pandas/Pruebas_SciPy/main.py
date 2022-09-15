import scipy
import numpy as np
"""
    Hankel orden 2 grado n - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel2.html#scipy.special.hankel2
"""
Hankel_orden_2 = scipy.special.hankel2(5,(2+8j), out=None)
print(Hankel_orden_2)

#https://github.com/scipy/scipy/issues/7722
def spherical_hn2(n,z,derivative=False):
    """ Spherical Hankel Function of the Second Kind """
    return scipy.special.spherical_jn(n,z,derivative=False)-1j*scipy.special.spherical_yn(n,z,derivative=False)

print(spherical_hn2(2,4))