import numpy as np

array = [1,2,3,0,4,2,1]

Array_fft = np.fft.fft(array)
print("El array tras FFT es",Array_fft)

Array_IFFT = np.abs(np.fft.ifft(Array_fft))

print("\n Array tras ifft:",Array_IFFT)