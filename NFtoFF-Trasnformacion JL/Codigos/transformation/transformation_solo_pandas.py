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

    # Definir las cabeceras manualmente
    column_names = ['x', 'y', 'z', 'Evalue']

    files_directory = flow_config['directory']

    # Delete of the normE field
    flow_config['files_in_directory'].pop()

    index = 0
    for file in flow_config['files_in_directory']:
        file_path = f"{files_directory}/{file}"
    
        # Leer el archivo, saltando las líneas que comienzan con '%'
        data = read_csv(file_path, comment='%', delim_whitespace=True, names=column_names, skiprows=1)
        
        x_coordinates = np.ma.masked_invalid(data['x'].to_numpy())
        y_coordinates = np.ma.masked_invalid(data['y'].to_numpy())
        z_coordinates = np.ma.masked_invalid(data['z'].to_numpy())

        fields['field_readed_masked']['coordinates'][file_type[index]] = {}
        fields['field_readed_masked']['coordinates'][file_type[index]]['x_coordinates'] = x_coordinates
        fields['field_readed_masked']['coordinates'][file_type[index]]['y_coordinates'] = y_coordinates
        fields['field_readed_masked']['coordinates'][file_type[index]]['z_coordinates'] = z_coordinates

        index += 1

def mask_values(fields, datatypes):
    for datatype in datatypes:
        field_component_with_no_nan = np.nan_to_num(fields['field_readed'][datatype], nan=0)
        fields['field_readed_masked'][datatype] = field_component_with_no_nan[np.nonzero(field_component_with_no_nan)]

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

def generate_points_to_interpolate(r_grid: object, theta_grid: object, phi_grid: object):
    """Function used to buils an array with the points to interpolate"""
    return np.column_stack((r_grid, theta_grid, phi_grid))

def change_versor_coordinates_to_spherical(Ex_interpolated: object, Ey_interpolated: object, Ez_interpolated: object , theta_grid: object, phi_grid: object):
    """Function used to change the versor coordinates to spherical from cartesians"""
    # Calcular componentes del vector en coordenadas esféricas
    E_r = Ex_interpolated * np.sin(theta_grid) * np.cos(phi_grid) + Ey_interpolated * np.sin(theta_grid) * np.sin(phi_grid) + Ez_interpolated * np.cos(theta_grid)
    E_theta = Ez_interpolated * np.cos(theta_grid) * np.cos(phi_grid) + Ey_interpolated * np.cos(theta_grid) * np.sin(phi_grid) - Ez_interpolated * np.sin(theta_grid)
    E_phi = -Ez_interpolated * np.sin(phi_grid) + Ey_interpolated * np.cos(phi_grid)
    
    return E_r, E_theta, E_phi

def translate_spherical_values_to_cartesians(r_grid: object, theta_grid: object, phi_grid: object):
    """Function responsible for translating spherical coordinates to Cartesian coordinates"""
    x_spherical = r_grid*np.sin(theta_grid)*np.cos(phi_grid)
    y_spherical = r_grid*np.sin(theta_grid)*np.sin(phi_grid)
    z_spherical = r_grid*np.cos(theta_grid)

    return x_spherical, y_spherical, z_spherical


if __name__ == '__main__':

    #try:

    fields = input_data_process(flow_config)
    
    read_data(fields)    

    # Este método quita de en medio los NaN.
    # mask_values(fields, fields['file_type'])

    E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid = change_coordinate_system_to_spherical(fields)

    near_field_to_far_field_transformation(E_r, E_theta, E_phi, r_grid, theta_grid, phi_grid)
    
    print("FIN PROGRAMA")
 