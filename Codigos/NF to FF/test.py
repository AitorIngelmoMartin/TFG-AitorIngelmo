import numpy as np
from scipy.interpolate import RegularGridInterpolator

dic = {
    "files": {
      "Ex": "c:\\\\Users\\\\aitor\\\\OneDrive\\\\Desktop\\\\TFG-AitorIngelmo\\\\Codigos\\\\NF to NF\\\\codigo JL//microstrip_patch_antenna_Ex.txt",
      "Ey": "c:\\\\Users\\\\aitor\\\\OneDrive\\\\Desktop\\\\TFG-AitorIngelmo\\\\Codigos\\\\NF to NF\\\\codigo JL//microstrip_patch_antenna_Ey.txt",
      "Ez": "c:\\\\Users\\\\aitor\\\\OneDrive\\\\Desktop\\\\TFG-AitorIngelmo\\\\Codigos\\\\NF to NF\\\\codigo JL//microstrip_patch_antenna_Ez.txt",
      "normE": "c:\\\\Users\\\\aitor\\\\OneDrive\\\\Desktop\\\\TFG-AitorIngelmo\\\\Codigos\\\\NF to NF\\\\codigo JL//microstrip_patch_antenna_normE.txt"
    },
    "file_type": [
      "Ex",
      "Ey",
      "Ez",
      "normE"
    ]
}
for file in dic['files'].keys(): #file_type
    filename = dic['files'][file]  # Nombre del archivo a leer
    print(filename)