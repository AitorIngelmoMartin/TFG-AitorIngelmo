header = """% Model:              microstrip_patch_antenna.mph
% Version:            COMSOL 5.0.0.243
% Date:               Mar 31 2023, 16:50
% Dimension:          3
% Nodes:              1000000
% Expressions:        1
% Description:        Electric field, y component
% Length unit:        mm
% Theta                       Phi                        r
"""
with open('python_FF_normR.txt', 'w') as f:
    f.write(header)

# Número entero
numero_entero = 5

# Convertir el número entero a float y formatearlo con 2 decimales
numero_formateado = "{:.2f}".format(numero_entero)
print(numero_formateado)  # Salida: 5.00
