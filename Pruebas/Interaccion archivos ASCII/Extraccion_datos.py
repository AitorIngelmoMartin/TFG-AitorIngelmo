"""
        Este pequeño código extrae los valores del archivo ASCII.
        Como pega, no es capaz de "adivinar" el valor de las columnas,
        por lo que está fijo para que empieze a partir de la linea
        12.
        Es decir, solo me vale para un tipo de archivo en concreto.
        De todas formas, no creo que haya mucho problema ya que los
        archivos de salida tienen la mísma estructura siempre.
"""
from astropy.io import ascii

data = ascii.read("archivo.ASC",converters={'': int}, data_start=12)
theta = Phi = E_real_PH = E_imag_PH  = E_real_PV = E_imag_PV =[] 

theta = data["col1"]
Phi   = data["col2"]
E_real_PH = data["col3"]
E_imag_PH = data["col4"]
E_real_PV = data["col5"]
E_imag_PV = data["col6"]