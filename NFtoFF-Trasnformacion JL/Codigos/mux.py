# import pandas as pd
# import csv

# # # field_names= ['n','m=-5', 'm=-4', 'm=-3', 'm=-2', 'm=-1', 'm=0', 'm=1', 'm=2', 'm=3', 'm=4', 'm=5']
# field_names= ['n','m=-5', 'm=-4', 'm=-3', 'm=-2', 'm=-1', 'm=0', 'm=1']

# amnffcoef_from_gmn = [
#     {
#         "n": 1, 
#         "m=-5": 0, 
#         "m=-4": 0, 
#         "m=-3": 0, 
#         "m=-2": 0, 
#         "m=-1": -0.009484061533268234+0.008122318008372756j, 
#         "m=0": 0.0, 
#         "m=1":-0.009484061533268239+0.008122318008372749j
#     }
# ]

# # # # convert np.arrays into dataframes
# # # amnffcoef_from_gmn_df = pd.DataFrame(amnffcoef_from_gmn)

# # # # save the dataframes as a csv files
# # # amnffcoef_from_gmn_df.to_csv(f"amnffcoef_from_gmn_calculated.csv")

# with open(f"amnffcoef_from_gmn_calculated.csv", "w", newline="") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=field_names)
#     writer.writeheader()
#     writer.writerows(amnffcoef_from_gmn)

# field_names= ['No', 'Company', 'Car Model']

# cars = [
#     {"No": 1, "Company": "Ferrari", "Car Model": "488 GTB"},
#     {"No": 2, "Company": "Porsche", "Car Model": "918 Spyder"},
#     {"No": 3, "Company": "Bugatti", "Car Model": "La Voiture Noire"},
#     {"No": 4, "Company": "Rolls Royce", "Car Model": "Phantom"},
#     {"No": 5, "Company": "BMW", "Car Model": "BMW X7"},
# ]

# # with open('Names.csv', 'w') as csvfile:
# #     writer = csv.DictWriter(csvfile, fieldnames=field_names)
# #     writer.writeheader()
# #     writer.writerows(cars)

# Datos a guardar

# Definir los datos complejos
data = [
    [(-0.009484061533268234+0.008122318008372756j), 0.0, (-0.009484061533268239+0.008122318008372749j)],
    [(-0.0016652721737404487+0.0013942088587150156j), (-0.001778615302666982+0.0014891026526039227j), 0.0, (-0.0017786153026669838+0.0014891026526039214j), (0.0016652721737404476-0.001394208858715018j)],
    [(-0.00013155522533815265+0.00011012250860670187j), (-0.00019265309369180007+0.00016126643326899644j), (-0.00011473695020549277+9.604423354552053e-05j), 0.0, (-0.00011473695020549267+9.604423354552046e-05j), (0.0001926530936918-0.00016126643326899674j), (-0.00013155522533815257+0.0001101225086067021j)],
    [(-6.742180433498912e-06+5.643754700633718e-06j), (-1.1572587562285198e-05+9.687199281797875e-06j), (-9.781263425396204e-06+8.187714935799833e-06j), (-3.7000170367048654e-06+3.09721591542917e-06j), 0.0, (-3.7000170367048666e-06+3.097215915429166e-06j), (9.781263425396192e-06-8.187714935799841e-06j), (-1.1572587562285186e-05+9.6871992817979e-06j), (6.74218043349891e-06-5.643754700633718e-06j)],
    [(-2.5607532232000537e-07+2.1435592193900312e-07j), (-4.888211636218883e-07+4.0918316627373284e-07j), (-4.953618733500669e-07+4.146582735633257e-07j), (-2.7072818373993713e-07+2.2662156155729087e-07j), (-8.024248332897942e-08+6.716950050799489e-08j), 0.0, (-8.024248332897951e-08+6.71695005079948e-08j), (2.7072818373993687e-07-2.2662156155729122e-07j), (-4.953618733500664e-07+4.146582735633262e-07j), (4.888211636218881e-07-4.0918316627373274e-07j), (-2.5607532232000447e-07+2.1435592193900386e-07j)]
]

# Función para guardar los datos en un archivo
def guardar_datos_complejos(file_name, data):
    n_id = 1
    header = '''% Valor guardado: Coeficiente Amn \n% Description: Coeficiente Amn a partir del cual podemos calcular el campo eléctrico en otros puntos\n'''
    with open(file_name, 'w',encoding='utf-8') as file:
        # Iterar sobre cada fila del array
        file.write(header)
        for row in data:
            # Convertir cada elemento a string, formateando los números complejos
            row_str = '   '.join(
                f'{x.real:.17g}+{x.imag:.17g}j' if isinstance(x, complex) else f'{x:.17g}'
                for x in row
            )
            # Escribir la fila formateada en el archivo
            file.write(f"n={n_id}\t\t\t{row_str}\n")
            n_id += 1

# Guardar los datos en un archivo de texto
guardar_datos_complejos('datos_complejos.txt', data)

print("Archivo con datos complejos guardado con éxito.")