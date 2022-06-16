# Este breve programa emplea la nomenclatura de la foto "triangulo referencia" para calcular los ángulos de
# la cámara.

import math

altura_antena_transmisora = 2.5
distancia_antena_medida   = 4

angulo_inferior = math.acos(altura_antena_transmisora / distancia_antena_medida)

distancia_pico_transmisor_suelo = altura_antena_transmisora / math.sin(angulo_inferior)

print("La distancia del pico del transmisor al suelo es:", distancia_pico_transmisor_suelo)