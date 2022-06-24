from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout


class Aplicacion(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        #Margenes
        self.window.size_hint = (0.6,0.7)
        self.window.pos_hint = {"center_x":0.5, "center_y":0.5}

        #Agregar imagene
        self.window.add_widget(Image(source="logo_uah.png"))
        #Widget label
        self.greeting = Label(
            text="Cualquier texto",
            font_size=18,
            color = "#00aae4"
            )

        self.window.add_widget(self.greeting)
        #Inputs
        self.user = TextInput(
                    multiline=False,
                    padding_y = (20,20),
                    size_hint = (1,0.5)
                    )
        self.window.add_widget(self.user)
        #Boton
        self.button = Button(
                      text="Nombre boton",
                      size_hint = (1,0.5),
                      bold = True,
                      background_color = "#00aae4",
                      background_normal = ""
                      )
        self.button.bind(on_press=self.callback)
        self.window.add_widget(self.button)

        return self.window
    def callback(self, instance):   
        self.greeting.text = "Has escrito: " + self.user.text + " como input"    

class maquetado(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    pass       
        

if __name__ == '__main__':
    Aplicacion().run()