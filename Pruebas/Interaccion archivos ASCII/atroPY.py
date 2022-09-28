from astropy.io import ascii

data = ascii.read("archivo.ASC",converters={'': int}, data_start=12)



print(data)

print(data[0])
