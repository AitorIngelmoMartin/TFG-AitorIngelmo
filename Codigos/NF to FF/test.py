import numpy as np
from scipy.interpolate import RegularGridInterpolator
mu0           = 4*np.pi*1e-7
c0            = 299792458
raw_data      = 9
pauseinterval = 0.01
k0 ,delta_x,L_x,delta_y,L_y,coordz = 0,0,0,0,0,0

def nearfieldPoint0toPoint1(fields,cut0,cut1):

    Nx = fields['zValueZeroedplane']['Ex'][cut0].shape[0]
    Ny = fields['zValueZeroedplane']['Ex'][cut0].shape[1]

    delta_kx  = 2*np.pi/(delta_x*Nx)
    delta_ky  = 2*np.pi/(delta_y*Ny) 

    factorf   = L_x*L_y*Nx*Ny/(4*np.pi**2*(Nx-1)*(Ny-1))        
    dimension = Nx

    #AGREGADO LO VISTO CON JLAP
    kx_array = np.arange(-Nx/2,Nx/2)*delta_kx #jlap
    ky_array = np.arange(-Ny/2,Ny/2)*delta_kx #jlap          
    #FIN AGREGACIÓN

    # Generación de los valores de los ángulos
    theta = np.linspace(0, 2 * np.pi, dimension)
    phi   = np.linspace(0, 2 * np.pi, dimension)
    # Generación de la malla
    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    
    #Cálculo de los Kx, Ky y Kz 
    kx = k0*np.sin(theta_mesh)*np.cos(phi_mesh)
    ky = k0*np.sin(theta_mesh)*np.sin(phi_mesh)
    kz = k0*np.cos(theta_mesh)

    
    kxy = np.array([[kx[i,j],ky[i,j]] for i in range(kx.shape[0]) for j in range(ky.shape[0])]).reshape(Nx,Nx,2)

    fields.update({'fields_transformed':{}})
    #Estos valores corresponden a la "Ad(kx,ky)"
    fields_to_transform = fields['zValueZeroedplane'].keys()
    for i in range(len(fields['zValueZeroedplane'].keys())):
        
        Ehat_component = factorf*np.fft.fft2((fields['zValueZeroedplane'][fields_to_transform[i]][cut0]))     
        Ehat_component_interp_func = RegularGridInterpolator((kx_array, ky_array), Ehat_component)
        Ehat_component_interp_data_func = Ehat_component_interp_func(kxy)

        factorMultiplicativo = (1j*kz*np.exp(-1j*k0))/(2*np.pi)
        Ehat_component_reconstruido = factorMultiplicativo*(Ehat_component_interp_data_func)
        fields['fields_transformed'].update({f"FF_{fields_to_transform[i]}":Ehat_component_reconstruido})
    
    #TODO
    Ehaty = factorf*np.fft.fft2((fields['zValueZeroedplane']['Ey'][cut0]))
    Ehatz = factorf*np.fft.fft2((fields['zValueZeroedplane']['Ez'][cut0]))
