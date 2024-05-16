"""File used to test the orthogonality of our functions"""
import funciones


def test_z1_matrix(N):
    """Function used to test the orthogonality of Z1"""
    total_result = []
    for n in range(1, N + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            value_calculated.append(funciones.z1(m,n))
        total_result.append(value_calculated)
    print(total_result)







