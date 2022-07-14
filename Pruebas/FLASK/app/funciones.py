import os 

def limpiar_directorio(directorio):
    """ Función que limpia el directorio de archivos. Lo uso para tener inicios limpios siempre """
    for archivo in os.listdir(directorio):
     os.remove(os.path.join(directorio, archivo))



