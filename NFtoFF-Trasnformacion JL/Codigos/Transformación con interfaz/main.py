"""Fichero encargado de soportar la interfaz"""
import json, transformation, logging, queue, threading, os
import tkinter as tk
from pathlib import Path
from datetime import datetime
from pandas import DataFrame
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog, messagebox, ttk, Menu, Text 

# Global variables
files_loaded = {}
file_names = []
exit_directory = Path(__file__).parent.resolve()
current_directory = Path(__file__).parent.resolve()
exit_directory_label = None
filemenu = None

# Logs config
logger = logging.getLogger('logger_maestro')
Path(f"{current_directory}/log_files/").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f"{current_directory}/log_files/log_file_from {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log", 
                encoding='utf-8', 
                datefmt='%m/%d/%Y %I:%M:%S %p',
                level=logging.DEBUG)

class QueueHandler(logging.Handler):
    """Class to send logging records to a queue

    It can be used from different threads
    The ConsoleUi class polls this queue to display records in a ScrolledText widget
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class ConsoleUi:
    """Poll messages from a logging queue and display them in a scrolled text widget"""

    def __init__(self, frame):
        self.frame = frame
        # Create a ScrolledText wdiget
        self.scrolled_text = ScrolledText(frame, state='disabled', height=12, bg='black', fg='white')        
        self.scrolled_text.pack(fill=tk.X, padx=0, pady=25)
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('INFO', foreground='white')
        self.scrolled_text.tag_config('DEBUG', foreground='gray')
        self.scrolled_text.tag_config('WARNING', foreground='orange')
        self.scrolled_text.tag_config('ERROR', foreground='red')

        # Create a logging handler using a queue
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self.frame.after(100, self.poll_log_queue)

    def display(self, record):
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, msg + '\n\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        # Autoscroll to the bottom
        self.scrolled_text.yview(tk.END)

    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self.frame.after(100, self.poll_log_queue)

def config_root_window(root):
    """Function used to config the root window"""
    # Add a title
    root.title("Convertidor campo cercano a campo lejano en esféricas")
    # Adjust size
    root.geometry("600x310")

def read_json_file():
    """Function used to open a file from a tkinter button"""
    global files_loaded
    
    file_path = filedialog.askopenfilename(
        title="Selecciona un fichero de configuración", filetypes=[("Text files", "*.json")])
    if file_path:
        with open(file_path, 'r') as file:
            file_name = file_path.split('/')[-1]
            file_names.append(file_name)
            files_loaded.update({
                file_name: json.load(file)
            })
            combo.configure(values=file_names)
            messagebox.showinfo(message=f"File {file_path} loaded", title="Info")
            logger.info(f'Se ha cargado el fichero {file_path} correctamente')
            filemenu.entryconfig("Procesar fichero", state= 'active')

def create_label():
    """Function used to create a label"""
    label = ttk.Label(text="Ficheros cargados:")
    label.pack(fill=tk.X, padx=5, pady=10)

def create_empty_combobox():
    """Function used to create the combobox that save the loaded files"""
    # Creamos un combo para mostrar los ficheros cargados
    combo = ttk.Combobox(
        state="readonly",
        width=30,
        values=[]
    )
    combo.place(x=110, y=10)

    return combo

def process_file():
    """Function used to process the selected file"""
    if combo.current() != -1:
        logger.info(f'Iniciando el procesamiento del fichero {file_names[combo.current()]}')
        file_to_process = files_loaded[f'{file_names[combo.current()]}']
        logger.info(f'El contenido del fichero es: {file_to_process}')
        
        amnffcoef_from_gmn, bmnffcoef_from_emn, far_field_calculated = transformation.main(file_to_process)
        save_results_in_file(amnffcoef_from_gmn, bmnffcoef_from_emn, far_field_calculated)

def save_results_in_file(amnffcoef_from_gmn, bmnffcoef_from_emn, far_field_calculated):
    """Function used to save the results obtained from a processed file"""

    # Check if the directory exists and create it if it doesn
    folder = f"outputs/{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}"
    if not os.path.exists(f"{exit_directory}/{folder}"):
        os.makedirs(f"{exit_directory}/{folder}")    
    
    # save the dataframes as a csv files
    save_data_to_file(f"{exit_directory}/{folder}/amnffcoef_from_gmn_calculated.txt",amnffcoef_from_gmn,coefficient='Amn')
    save_data_to_file(f"{exit_directory}/{folder}/bmnffcoef_from_emn_calculated.txt",bmnffcoef_from_emn,coefficient='Bmn')

    e_components = ["Ex","Ey","Ez"]
    for i in range(3):
        far_field_calculated_df = DataFrame(far_field_calculated[:,:,i])
        far_field_calculated_df.to_csv(f"{exit_directory}/{folder}/far_field_calculated_{e_components[i]}.csv")

    logger.info(f'Resultados guardados correctamente en el directorio {exit_directory}/{folder}')

def save_data_to_file(file_name, data, coefficient):
    """Function used to save the results on files"""
    n_id = 1
    header = f'''% Valor guardado: Coeficiente {coefficient} \n% Description: Coeficiente Amn a partir del cual podemos calcular el campo eléctrico en otros puntos\n'''
    with open(file_name, 'w',encoding='utf-8') as file:
        # Write the header of the file
        file.write(header)
        for row in data:
            # Fomating the complex numbers
            row_str = '   '.join(
                f'{x.real:.17g}+{x.imag:.17g}j' if isinstance(x, complex) else f'{x:.17g}'
                for x in row
            )
            # write the row on the file
            file.write(f"n={n_id}\t\t\t{row_str}\n")
            n_id += 1

def modify_exit_directory():
    """Function used to modify the exit directory"""
    global exit_directory, exit_directory_label
    exit_directory = filedialog.askdirectory()

    # Añadimos un label que muestra cual es el directorio
    if exit_directory_label is None:
        exit_directory_label = ttk.Label(text=f"El directorio de trabajo actual es: {exit_directory}")
        exit_directory_label.pack(fill=tk.X, padx=0, pady=0)
    else:
        exit_directory_label["text"]=f"El directorio de trabajo actual es: {exit_directory}"

def open_info_window():
    """Function used to create a secondary windows that have some info about the GUI"""
    # Create secondary (or popup) window.
    secondary_window = tk.Toplevel()
    secondary_window.title("Información sobre la interfaz")
    secondary_window.config(width=400, height=400)

    # Create text widget and specify size.
    T = Text(secondary_window, wrap=tk.WORD)

    # Create label
    l = ttk.Label(secondary_window, text = "Sobre la interfaz")
    l.config(font =("Courier", 14))

    documentation = """Esta interfaz pretende, de forma básica y simple, poder interactuar con el código que nos permite realizar la transformada.\n
    \nPara ello, en la ventana inicial podemos cargar ficheros de configuración desde la opción 'Archivo' para luego procesarlos una vez seleccionados en el combo.\n
    \nTras haberlos procesado, en el directorio en el que se ejecute la aplicación se generará una carpeta 'outputs' y dentro de ella una carpeta con la fecha actual donde almacenará los resultados.\n
    \nSi queremos modificar el directorio donde se dejan los resultados, podemos hacerlo desde el botón 'Editar'. En caso de modificarlo, aparecerá en texto plano el directorio que estamos usando para saber dónde buscar los resultados."""

    l.pack()
    T.pack(side=tk.LEFT, fill=tk.Y)

    # Insert The Fact.
    T.insert(tk.END, documentation)

    # Create a button to close (destroy) this window.
    button_close = ttk.Button(
        secondary_window,
        text="Cerrar ventana",
        command=secondary_window.destroy
    )

def create_menu(root):
    """Function used to create our menu"""
    global filemenu
    menubar = Menu(root)
    root.config(menu=menubar)

    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="Cargar fichero", command=read_json_file)
    filemenu.add_command(label="Procesar fichero", state="disabled",command=threading.Thread(target=process_file).start)
    filemenu.add_separator()
    filemenu.add_command(label="Salir", command=root.quit)

    editmenu = Menu(menubar, tearoff=0)
    editmenu.add_command(label="Seleccionar directorio de salida", command=modify_exit_directory)

    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Uso de la interfaz", command=open_info_window)

    menubar.add_cascade(label="Archivo", menu=filemenu)
    menubar.add_cascade(label="Editar", menu=editmenu)
    menubar.add_cascade(label="Ayuda", menu=helpmenu)


if __name__ == '__main__':
    root = tk.Tk()
    
    config_root_window(root)
    
    create_menu(root)

    create_label()

    combo = create_empty_combobox()

    scrolledText = ConsoleUi(root)

    # Bucle principal de Tkinter
    root.mainloop()
