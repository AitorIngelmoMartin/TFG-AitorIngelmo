"""File used to perform the NF2FF transformation"""
import os, sys
import numpy as np
import numpy.ma as ma
from math import cos, sin
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


# Import 'funciones' module
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import funciones

#VARIABLES GLOBALES
mu0           = 4*np.pi*1e-7
c0            = 299792458
raw_data      = 9
pauseinterval = 0.01
threshold     = 1e-10
delta_x, delta_y = 0 , 0
file_type = ['Ex','Ey','Ez','normE']
k_calculated = 0

flow_config = { 
    'directory':r'c:\\Users\\aitor\\OneDrive\\Desktop\\TFG-AitorIngelmo\\NFtoFF-Trasnformacion JL\\Codigos\\transformation',
    'files_in_directory': [
        'microstrip_patch_antenna_Ex.txt',
        'microstrip_patch_antenna_Ey.txt',
        'microstrip_patch_antenna_Ez.txt',
        'microstrip_patch_antenna_normE.txt'                
    ],
    'file_type':file_type,
    'work_mode':'NFtoFF',      
    'shape':[100,100,100],
    'freq':1.575e9,
    'length_unit':1e-3,
    'res':2e-3
}

def input_data_process(flow_config: dict):
    """Function used to save inputs from our flow_config dictionary"""
    global k_calculated
    try:
        fields,files = {}, {}
        if len(flow_config['file_type']) == len(flow_config['files_in_directory']):
            for file_type in range(len(flow_config['file_type'])):
                files.update({flow_config['file_type'][file_type]:f"{flow_config['directory']}//{flow_config['files_in_directory'][file_type]}"})
        else:
            raise Exception("El número de ficheros en el directorio es distinto a la cantidad de ficheros introducido")
        fields.update({'files':files})
        fields.update({'file_type':flow_config['file_type']})
        fields.update({"field_readed":{}})
        fields.update({"lines":{}})
        fields.update({"field_readed_masked":{}})
        fields.update({'fields_transformed':{}})
        fields.update({'quantitative_comparison':{}})
        fields.update({'freq':flow_config['freq']})
        fields.update({'length_unit':flow_config['length_unit']})
        fields.update({'res':flow_config['res']})

        k_calculated = (2*np.pi * flow_config['freq'])/c0

    except Exception as exc:
        print(f"ERROR:{exc}")

    return fields

def read_data(fields):
    """Function used to read data from the files defined in 'flow_config'"""
    # Definir las cabeceras manualmente
    column_names = ['x', 'y', 'z', 'E_value']

    fields['field_readed_masked']['values_readed'] = {}
    for file_type, file_path in fields['files'].items():
        with open(file_path) as file:
            fields['lines'][file_type] = file.readlines()

            # Leer el archivo, saltando las líneas que comienzan con '%'
            data = read_csv(file_path, comment='%', delim_whitespace=True, names=column_names, skiprows=1)
            
            data_with_no_nans = data.dropna()

            x_coordinates = data_with_no_nans['x'].to_numpy()
            y_coordinates = data_with_no_nans['y'].to_numpy()
            z_coordinates = data_with_no_nans['z'].to_numpy()
            E_value = data_with_no_nans['E_value'].to_numpy()

            E_value_str = np.array([str(val) for val in E_value])
            E_value_j = np.char.replace(E_value_str , 'i', 'j')
            complex_E_value = E_value_j.astype(complex)

            fields['field_readed_masked']['values_readed'][file_type] = {}
            fields['field_readed_masked']['values_readed'][file_type]['x_coordinates'] = x_coordinates
            fields['field_readed_masked']['values_readed'][file_type]['y_coordinates'] = y_coordinates
            fields['field_readed_masked']['values_readed'][file_type]['z_coordinates'] = z_coordinates
            fields['field_readed_masked']['values_readed'][file_type]['E_value'] = complex_E_value

def extract_matrix_data(fields):
    """
    Este método almacena en arrays los datos de los ficheros de Comsol leídos previamente con readData             
    """
    global delta_x, delta_y
    #filetypes = lines.keys()
    coord_flag = None
    for file_type in fields['lines'].keys(): #file_type
        datatype     = file_type
        rawdatalines = fields['lines'][file_type][raw_data:]
        if coord_flag == None:
            coord_flag = 1
            coord      = np.array([[float(s) for k,s in zip(range(4),rawdatalines[i].split()) if k<3 ] \
                for i in range(len(rawdatalines))]).reshape(len(rawdatalines),3)
            
            coordx  = np.unique(coord[:,0])*fields['length_unit']
            delta_x = coordx[1]-coordx[0]

            coordy  = np.unique(coord[:,1])*fields['length_unit']
            delta_y = coordy[1]-coordy[0]

        fields['field_readed'][datatype] = np.array([complex(s.replace('i', 'j')) for i in range(len(rawdatalines)) \
            for k,s in zip(range(4),rawdatalines[i].split()) if k == 3])

def extract_coordinates(flow_config: dict, fields: dict):
    """Function used to extract coordinates from our files"""
    fields['field_readed_masked']['coordinates'] = {}

    # Definition of dataframe header names
    column_names = ['x', 'y', 'z', 'Evalue']

    files_directory = flow_config['directory']

    # Delete of the normE field read previously
    flow_config['files_in_directory'].pop()

    index = 0
    for file in flow_config['files_in_directory']:
        file_path = f"{files_directory}/{file}"
    
        # Read the file, skipping lines starting with '%' (Comsol headers)
        data = read_csv(file_path, comment='%', delim_whitespace=True, names=column_names, skiprows=1)
        
        x_coordinates = np.ma.masked_invalid(data['x'].to_numpy())
        y_coordinates = np.ma.masked_invalid(data['y'].to_numpy())
        z_coordinates = np.ma.masked_invalid(data['z'].to_numpy())

        fields['field_readed_masked']['coordinates'][file_type[index]] = {}
        fields['field_readed_masked']['coordinates'][file_type[index]]['x_coordinates'] = x_coordinates
        fields['field_readed_masked']['coordinates'][file_type[index]]['y_coordinates'] = y_coordinates
        fields['field_readed_masked']['coordinates'][file_type[index]]['z_coordinates'] = z_coordinates

        index += 1

def change_coordinate_system_to_spherical(fields: dict):
    """Function used to change the coordinate system from cartesians to sphericals"""

    r_grid, theta_grid, phi_grid = generate_spherical_values(theta_init=-np.pi/3,
                                              theta_end=np.pi/3,
                                              phi_init=0,
                                              phi_end=2*np.pi,
                                              number_of_values = 5,
                                              r=0.01)

    Ex_interpolator = make_interpolator(fields, 'Ex')
    Ey_interpolator = make_interpolator(fields, 'Ey')
    Ez_interpolator = make_interpolator(fields, 'Ez')

    x_spherical, y_spherical, z_spherical = translate_spherical_values_to_cartesians(r_grid, theta_grid, phi_grid)

    points_to_interpolate = generate_points_to_interpolate(x_spherical, y_spherical, z_spherical)

    Ex_interpolated = Ex_interpolator(points_to_interpolate)
    Ey_interpolated = Ey_interpolator(points_to_interpolate)
    Ez_interpolated = Ez_interpolator(points_to_interpolate)

    E_r, E_theta, E_phi = change_versor_coordinates_to_spherical(Ex_interpolated, Ey_interpolated, Ez_interpolated, theta_grid, phi_grid)

    return E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid

def generate_spherical_values(theta_init: int, theta_end: int, phi_init: int, phi_end: int, number_of_values: int, r: int = 1):
    """Function used to generate a spherical grid"""
    
    # Creation of base coordinates
    theta = np.linspace(theta_init, theta_end, number_of_values)
    phi = np.linspace(phi_init, phi_end, number_of_values)
    
    r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')

    return r_grid.flatten(), theta_grid.flatten(), phi_grid.flatten()
    
def make_interpolator(fields: dict, field_component: str):
    """Function used to make an interpolator to obtain spherical values"""
    x_coordenates = fields['field_readed_masked']['values_readed'][field_component]['x_coordinates']
    y_coordinates = fields['field_readed_masked']['values_readed'][field_component]['y_coordinates']
    z_coordinates = fields['field_readed_masked']['values_readed'][field_component]['z_coordinates']
    
    measure_points_stack = np.column_stack((x_coordenates, y_coordinates, z_coordinates))
    Evalue = fields['field_readed_masked']['values_readed'][field_component]['E_value']

    # Creation of the interpolator
    linear_interpolador = LinearNDInterpolator(points=measure_points_stack, values=Evalue)

    return linear_interpolador

def generate_points_to_interpolate(r_grid: list, theta_grid: list, phi_grid: list):
    """Function used to buils an array with the points to interpolate"""
    return np.column_stack((r_grid, theta_grid, phi_grid))

def change_versor_coordinates_to_spherical(Ex_interpolated: list, Ey_interpolated: list, Ez_interpolated: list , theta_grid: list, phi_grid: list):
    """Function used to change the versor coordinates to spherical from cartesians"""
    E_r     = Ex_interpolated * np.sin(theta_grid) * np.cos(phi_grid) + Ey_interpolated * np.sin(theta_grid) * np.sin(phi_grid) + Ez_interpolated * np.cos(theta_grid)
    E_theta = Ez_interpolated * np.cos(theta_grid) * np.cos(phi_grid) + Ey_interpolated * np.cos(theta_grid) * np.sin(phi_grid) - Ez_interpolated * np.sin(theta_grid)
    E_phi   = -Ez_interpolated * np.sin(phi_grid) + Ey_interpolated * np.cos(phi_grid)
    
    return E_r, E_theta, E_phi

def translate_spherical_values_to_cartesians(r_grid: list, theta_grid: list, phi_grid: list):
    """Function responsible for translating spherical coordinates to Cartesian coordinates"""
    x_spherical = r_grid*np.sin(theta_grid)*np.cos(phi_grid)
    y_spherical = r_grid*np.sin(theta_grid)*np.sin(phi_grid)
    z_spherical = r_grid*np.cos(theta_grid)

    return x_spherical, y_spherical, z_spherical

def near_field_to_far_field_transformation(E_r: list, E_theta: list, E_phi: list , r_grid: list, theta_grid: list, phi_grid: list, number_of_modes: int):
    """Function used to transfor our near field mesures into far field ones"""

    delta_theta = calculate_measure_point_increments(theta_grid)
    delta_phi   = calculate_measure_point_increments(phi_grid)

    emn_calculated = calculate_emn(E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid,number_of_modes, delta_theta, delta_phi)
    gmn_calculated = calculate_gmn(E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid,number_of_modes, delta_theta, delta_phi)

    amnffcoef_from_gmn = calculate_amnffcoef_from_gmn(number_of_modes, gmn_calculated, r=r_grid[0], k=k_calculated)
    bmnffcoef_from_emn = calculate_bmnffcoef_from_emn(number_of_modes, emn_calculated, r=r_grid[0], k=k_calculated)
    
    return amnffcoef_from_gmn, bmnffcoef_from_emn

def calculate_measure_point_increments(vector):
    """Function used to get the increment of a regular vector"""
    return np.max(np.unique(np.diff(vector)))

def calculate_emn(E_r: list, E_theta: list, E_phi: list , r_grid: list, theta_grid: list, phi_grid: list, number_of_modes: int, delta_theta: int, delta_phi: int):
    """Function used to obtain a single value of emn from our Dipole field"""
    total_result = []
    threshold = 1e-10
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            emn_value = calculate_emn_value(
                E_r=E_r,
                E_theta=E_theta,
                E_phi=E_phi,
                r_grid=r_grid,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
                m=m,
                n=n,
                delta_theta=delta_theta,
                delta_phi=delta_theta)
            aux = sum(emn_value)
            if abs(aux) < threshold:
                aux = 0.0
            value_calculated.append(aux)
        total_result.append(value_calculated)
    return total_result

def calculate_emn_value(E_r: list, E_theta: list, E_phi: list, r_grid: list, theta_grid: list, phi_grid: list, m: int, n: int, delta_theta: int, delta_phi: int):
    """Function used to obtain a single value of emn from our Dipole field"""

    total_result = np.array([np.complex128(0),np.complex128(0),np.complex128(0)])
    theta_index = 0    
    for theta in theta_grid:
        phi_index = 0
        for phi in phi_grid:
            total_result += np.array([E_r[0],E_theta[theta_index],E_phi[phi_index]])*funciones.b_sin_function(-m,n,theta,phi)
            phi_index += phi_index
        theta_index += theta_index
    return total_result*delta_theta*delta_phi

def calculate_gmn(E_r: list, E_theta: list, E_phi: list , r_grid: list, theta_grid: list, phi_grid: list, number_of_modes: int, delta_theta: int, delta_phi: int):
    """Function used to obtain a single value of gmn from our Dipole field"""
    total_result = []
    threshold = 1e-10
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            gmn_value = calculate_gmn_value(
                E_r=E_r,
                E_theta=E_theta,
                E_phi=E_phi,
                r_grid=r_grid,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
                m=m,
                n=n,
                delta_theta=delta_theta,
                delta_phi=delta_theta)
            aux = sum(gmn_value)
            if abs(aux) < threshold:
                aux = 0.0
            value_calculated.append(aux)
        total_result.append(value_calculated)
    return total_result

def calculate_gmn_value(E_r: list, E_theta: list, E_phi: list, r_grid: list, theta_grid: list, phi_grid: list, m: int, n: int, delta_theta: int, delta_phi: int):
    """Function used to obtain a single value of gmn from our Dipole field"""

    total_result = np.array([np.complex128(0),np.complex128(0),np.complex128(0)])
    theta_index = 0    
    for theta in theta_grid:
        phi_index = 0
        for phi in phi_grid:
            total_result += np.array([E_r[0],E_theta[theta_index],E_phi[phi_index]])*funciones.c_sin_function(-m,n,theta,phi)
            phi_index += phi_index
        theta_index += theta_index
    return total_result*delta_theta*delta_phi

def calculate_bmnffcoef_from_emn(number_of_modes: int, emn: list, r: int, k: int):
    """Function used to obtain a all values of bmnffcoef from bmn"""
    total_result = []
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            b_coef_value = funciones.b_coef_function(e_data=emn,
                                           m=m,
                                           n=n,
                                           k=k,
                                           R=r)
            if abs(b_coef_value) < threshold:
                b_coef_value = 0.0
            value_calculated.append(b_coef_value)
        total_result.append(value_calculated)
    return total_result

def calculate_amnffcoef_from_gmn(number_of_modes: int, gmn: list, r: int, k: int):
    """Function used to obtain a all values of Amnffcoef from gmn"""
    total_result = []
    
    for n in range(1, number_of_modes + 1):
        value_calculated = []
        for m in range(-n, n + 1):
            b_coef_value = funciones.a_coef_function(g_data=gmn,
                                           m=m,
                                           n=n,
                                           k=k,
                                           R=r)
            if abs(b_coef_value) < threshold:
                b_coef_value = 0.0
            value_calculated.append(b_coef_value)
        total_result.append(value_calculated)
    return total_result

def calculate_far_field(number_of_modes: int, a_coef_value: list, b_coef_value: list, theta_values, phi_values):
    """Function used to obtainthe far field""" 
    far_field_calculated = np.ndarray((np.size(phi_values),np.size(theta_values),3), dtype=complex)
    theta_index = 0
    for theta in theta_values:
        phi_index = 0
        for phi in phi_values:
            far_field_value = calculate_far_field_value(number_of_modes, a_coef_value, b_coef_value, r=1, k=k_calculated, theta=theta, phi=phi)
            far_field_calculated[phi_index,theta_index,:] = far_field_value
            phi_index += 1
        theta_index += 1
    return far_field_calculated

def calculate_far_field_2D(number_of_modes: int, a_coef_value: list, b_coef_value: list, theta_values, phi_values):
    """Function used to obtainthe far field keeping theta or phi constant""" 
    far_field_calculated = []
    if type(phi_values) == int:
        for theta in theta_values:
            far_field_value = calculate_far_field_value(number_of_modes, a_coef_value, b_coef_value, r=1, k=k_calculated, theta=theta, phi=phi_values)
            far_field_calculated.append(far_field_value)
    elif type(theta_values) == int:
        for phi in phi_values:
            far_field_value = calculate_far_field_value(number_of_modes, a_coef_value, b_coef_value, r=1, k=k_calculated, theta=theta_values, phi=phi)
            far_field_calculated.append(far_field_value)

def calculate_far_field_value(number_of_modes: int, a_coef_value: list, b_coef_value: list, r: int, k: int, theta: int, phi: int):
    """Function used to obtainthe far field"""
    total_result = np.array([np.complex128(0),np.complex128(0),np.complex128(0)])
    for n in range(1, number_of_modes + 1):
        for m in range(-n, n + 1):
            total_result += ((2*n+1)/(n*(n+1)))*(a_coef_value[n-1][n+m]*funciones.m_function_far_field_aproximation(m,n,k,r,theta,phi)+b_coef_value[n-1][n+m]*funciones.n_function_far_field_aproximation(m,n,k,r,theta,phi))
    return total_result

def draw_radiation_diagram(thetha: list, module_far_field_calculated: list):
    """Function used to make plots to review the results"""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(thetha, module_far_field_calculated)
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()

def draw_module_diagram(far_field_calculated, theta_values, phi_values):
    """Function used to draw a 3D diagram of our field"""
    # Er, Etheta, Ephi = obtain_field_components(far_field_calculated) #(100,100,3)

    # r = 1.0
    theta, phi = np.meshgrid(theta_values, phi_values)
    theta_flatten = theta.flatten()
    phi_flatten = phi.flatten()

    Ex, Ey, Ez = change_coordinate_system_to_cartesian(far_field_calculated, theta, phi)

    # Obtain the module of the field
    Emodule = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
    R = Emodule

    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # Crear la figura y el eje 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.plot_surface(x, y, z, cmap=plt.get_cmap('jet'))

    # Mostrar la gráfica
    plt.show()

def obtain_field_components(far_field_calculated):
    """Function used to obtain each field component in separete arrays"""
    Er = np.array([])
    Etheta = np.array([])
    Ephi = np.array([])
    for value in far_field_calculated:
        Er = np.append(Er, value[0])
        Etheta = np.append(Etheta, value[1])
        Ephi = np.append(Ephi, value[2])

    return Er, Etheta, Ephi

def change_coordinate_system_to_cartesian(far_field_calculated: object, theta, phi):
    """Function used to change the coordinate system to cartesian from spherical
        far_field_calculated: Tensor de dimensiones 100,100,3 (shape de theta y phi) que contiene los valores del campo Er,Etheta,Ephi en
        su 3a di
    """

    Ex = far_field_calculated[:,:,0] * np.sin(theta) * np.cos(phi) + far_field_calculated[:,:,1] * np.cos(theta) * np.cos(phi) - far_field_calculated[:,:,2] * np.sin(phi)
    Ey = far_field_calculated[:,:,0] * np.sin(theta) * np.sin(phi) + far_field_calculated[:,:,1] * np.cos(theta) * np.sin(phi) + far_field_calculated[:,:,2] * np.cos(phi)
    Ez = far_field_calculated[:,:,0] * np.cos(theta) - far_field_calculated[:,:,1] * np.sin(theta)

    return Ex, Ey, Ez

if __name__ == '__main__':

    fields = input_data_process(flow_config)
    
    read_data(fields)    

    E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid = change_coordinate_system_to_spherical(fields)

    number_of_modes = 5
    amnffcoef_from_gmn, bmnffcoef_from_emn = near_field_to_far_field_transformation(E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid, number_of_modes)
    
    # Points where i want to know the field
    n = 100
    theta_values = np.linspace(0, np.pi, n)
    phi_values = np.linspace(0, 2 * np.pi, n)
    far_field_calculated = calculate_far_field(number_of_modes, amnffcoef_from_gmn, bmnffcoef_from_emn, theta_values, phi_values)
    draw_module_diagram(far_field_calculated,theta_values,phi_values)

    # # Points where i want to know the field
    # n = 100
    # theta_values = np.linspace(0, 2*np.pi, n)
    # far_field_calculated = calculate_far_field(number_of_modes, amnffcoef_from_gmn, bmnffcoef_from_emn, theta_values, phi_value=50)
    # module_far_field_calculated = np.linalg.norm(far_field_calculated, axis=1)
    # draw_radiation_diagram(theta_values, module_far_field_calculated)

    # # Points where i want to know the field
    # n = 100
    # phi_value = np.linspace(0, 2*np.pi, n)
    # far_field_calculated = calculate_far_field(number_of_modes, amnffcoef_from_gmn, bmnffcoef_from_emn, theta_values=50, phi_value=phi_value)
    # module_far_field_calculated = np.linalg.norm(far_field_calculated, axis=1)
    # draw_radiation_diagram(phi_value, module_far_field_calculated)

    print("FIN PROGRAMA")
