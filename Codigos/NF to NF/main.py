file  = open(r'Datos_antena\NF_30mmX.txt', 'r')

NF_30mmX = []
NF_30mmY = []
NF_40mmX = []
NF_40mmY = []   

lines   = file.readlines()
rawdata = lines[8:]

print(rawdata)


for i in range(len(rawdata)):
    columnas = rawdata[i].split()
    NF_30mmX.append(float(columnas[3]))


print(NF_30mmX[1])