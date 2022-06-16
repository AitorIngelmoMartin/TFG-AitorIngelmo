# Archivo empleado a modo de apoyo para la prueba de la aplicación
# Autor: @AitorIngelmoMartin
# Fecha inicio: 16/06/2022
# Fecha final: 

# make a function with read a document from a route and show the content
def read_document(route):
    with open(route, "r") as file:
        content = file.read()
        print(content)

ruta_documento = "Notas.txt"
read_document(ruta_documento)
