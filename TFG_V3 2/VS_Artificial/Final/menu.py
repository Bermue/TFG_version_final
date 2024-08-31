
# menu
# Menú encargado de abrir los demás archivos .py y unificar todo en el mismo programa

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from Calibrado import Calcular, MainWindow2
from CoordenadasDobleCam_V3 import ObjectDetection
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium



def archivo_Calibracion():
    exec(open('VS_Artificial/Final/Calibrado.py').read())
       

def archivo_Deteccion_objetos():
    exec(open('VS_Artificial/Final/CoordenadasDobleCam_V3.py').read())





class MenuWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Menú de Selección')
        self.setGeometry(100, 100, 300, 200)


        layout = QVBoxLayout()

        self.actualizar = QPushButton('Actualizar')
        self.actualizar.clicked.connect(self.actualizar_texto_calibracion)
        layout.addWidget(self.actualizar)


        self.label_calibrado = QLabel()
        self.label_baseline = QLabel()
        self.actualizar_texto_calibracion() 
        layout.addWidget(self.label_calibrado)
        layout.addWidget(self.label_baseline)

        
        


        button1 = QPushButton('Calibrado', self)
        button1.clicked.connect(archivo_Calibracion)
        button1.clicked.connect(self.abrir_ventana_Calibrado)
        layout.addWidget(button1)

        button2 = QPushButton('Live', self)
        button2.clicked.connect(archivo_Deteccion_objetos)
        layout.addWidget(button2)

        self.setLayout(layout)

    def abrir_ventana_Calibrado(self):
        calculos = Calcular()
        # Crear una instancia de la ventana de Calibrado.py
        self.archivo1_window = MainWindow2(calculos)  
        self.archivo1_window.show() 

    def leer_distancia_focal(self):
        try:
            with open("DB/Almacen/distanciaFocal.txt", 'r') as archivo:
                lineas = archivo.readlines()
                if lineas:
                    # Convertir el último número a flotante
                    ultimo_numero = float(lineas[-1].strip()) 
                    return ultimo_numero
                else:
                    return None
        except (FileNotFoundError, ValueError):
            return None
    
    def leer_baseline(self):
        try:
            with open("DB/Almacen/baseline.txt", 'r') as archivo:
                lineas = archivo.readlines()
                if lineas:
                    ultimo_numero = float(lineas[-1].strip()) 
                    return ultimo_numero
                else:
                    return None
        except (FileNotFoundError, ValueError):
            return None
        
    def actualizar_texto_calibracion(self):
        distancia_focal = self.leer_distancia_focal()
        baseline = self.leer_baseline()
        if baseline is not None:
            texto = f"Baseline actual: {baseline}"
            self.label_baseline.setText(texto)
            # Limpia cualquier estilo existente
            self.label_baseline.setStyleSheet("")  
        else:
            texto = "Baseline: Pendiente"
            self.label_baseline.setText(texto)
            self.label_baseline.setStyleSheet("color: red")  



        if distancia_focal is not None:
            texto = f"Calibración actual: {distancia_focal}"
            self.label_calibrado.setText(texto)
            # Limpia cualquier estilo existente
            self.label_calibrado.setStyleSheet("")  
        else:
            texto = "Calibración: Pendiente"
            self.label_calibrado.setText(texto)
            self.label_calibrado.setStyleSheet("color: red")  


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MenuWindow()
    window.show()
    sys.exit(app.exec_())
