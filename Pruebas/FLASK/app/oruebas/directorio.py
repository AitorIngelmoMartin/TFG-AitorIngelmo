import os

# folder path
dir_path = 'static/uploads'
numero_archivos_uploads = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        numero_archivos_uploads += 1
print('El numero de archivos subidos es:', numero_archivos_uploads)




import os
dir_path = 'static/uploads'
numero_archivos_uploads = 0
{% for path in os.listdir(dir_path) -%} <!-- CONTADOR ARCHIVOS  -->


  {% if os.path.isfile(os.path.join(dir_path, path)) %} 
    numero_archivos_uploads += 1
  {% endif %}
<p> El número de archivos subidos es: {{numero_archivos_uploads}}</p>
{% endfor -%}