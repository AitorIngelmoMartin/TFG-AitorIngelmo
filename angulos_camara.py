# Este breve programa emplea la nomenclatura de la foto "triangulo referencia" para calcular los ángulos de
# la cámara.
print("\n")
import math

altura_antena_transmisora = 30
distancia_antena_medida   = 30

def rad2deg(rad):
    return rad * 180 / math.pi

# Este valor es Alpha_H
angulo_inferior = math.atan(  altura_antena_transmisora   / distancia_antena_medida)
print("El ángulo inferior es: ", angulo_inferior, "radianes o ", rad2deg(angulo_inferior), "grados \n")

distancia_transmisor_suelo_receptor = altura_antena_transmisora / math.sin(angulo_inferior)
print("La distancia del pico del transmisor al suelo es:", distancia_transmisor_suelo_receptor,"\n")


# haz una funcion que pase de coordenadas polares a cilindricas
# make it
def pol2cart(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y
# y otra que pase de coordenadas cilindricas a polares
# make it
def cart2pol(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta



