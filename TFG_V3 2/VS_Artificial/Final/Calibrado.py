
# Calibrado
# Calcula la distancia focal en pixels según datos que introduzcas manualmente 
# O por foto, introduciendo la distancia al objeto y la linea base



import cv2
import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QStackedWidget


class Calcular:
    def __init__(self):
        #elegir si usar nucleos cuda o cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.disparidad= 0
        

    def _write(self, content):
        try:
            with open("DB/Almacen/log.txt", 'a') as archivo:
                archivo.write(content)
            print(f"Se ha escrito correctamente en el log.")
        except Exception as e:
            print(f"Error al escribir en el log: {e}")

    
    def load_model(self):
        model = YOLO("weights/best90.pt") 
        model.fuse()
        return model
    
    def predict(self, frame):
        results = self.model(frame)
        return results

    # Función para calcular la disparidad
    def calcular_disparidad(self, left_bbox, right_bbox):
        # Calcular el punto central del cuadro delimitador en ambas imágenes en el eje x
        centros_left = left_bbox[0]
        centro_x_left = centros_left[0]

        centros_right = right_bbox[0]
        centro_x_right = centros_right[0]


        # Calcular la disparidad (diferencia horizontal entre los centros)
        disparity = abs(centro_x_left - centro_x_right)
        return disparity


    # Función para mostrar rectángulos delimitadores
    def plot_boxes(self, results):
            class_ids = []
            centers = []

            for result in results:
                boxes = result.boxes.cpu().numpy()
                xyxyss  = boxes.xyxy

                

                for xyxy in xyxyss:
                    centro_x = (int(xyxy[0]) + int(xyxy[2]))/2
                    centro_y = (int(xyxy[1]) + int(xyxy[3]))/2
                    centers.append((centro_x, centro_y))

                class_ids.append(boxes.cls)
            
            
            return centers, class_ids


    def calculos(self):
        #Con el fotograma del video
        img_left = cv2.imread('DB/fotograma_1.png')
        img_right = cv2.imread('DB/fotograma_2.png')

        # Detectar objetos en la imagen izquierda
        results_left = self.predict(img_left)

        # Detectar objetos en la imagen derecha
        results_right = self.predict(img_right)

        # Obtener las coordenadas delimitadoras, confianzas, identificadores de clase y centros de los cuadros delimitadores para la izquierda
        centers_left, class_id = self.plot_boxes(results_left)

        # Obtener las coordenadas delimitadoras, confianzas, identificadores de clase y centros de los cuadros delimitadores para la derecha
        centers_right, class_id2 = self.plot_boxes(results_right)
        
        self._write(f'se ha detectado el objeto con id: {class_id[0][0]} en la foto1\n')
        self._write(f'se ha detectado el objeto con id: {class_id[0][0]} en la foto2\n')
        


        # Calcular profundidad para el primer objeto detectado en la izquierda
        self.disparidad = self.calcular_disparidad(centers_left, centers_right)
        self._write(f"Disparidad calculada: {self.disparidad}\n")



# Crear una ventana con PyQt para ingresar la profundidad
class MainWindow2(QWidget):
    def __init__(self, Calculos):
        super().__init__()
        self.calculos = Calculos
        self.model = Calculos.load_model()

        self.setWindowTitle('Calcular Distancia Focal')
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout()

        # Crear los botones "Arriba" y "Abajo"
        self.btn_arriba = QPushButton('Calibrado a foto', self)
        self.btn_abajo = QPushButton('Calibrado manual', self)
        self.btn_mas_abajo = QPushButton('Hacer foto', self)

        # Conectar los botones a sus funciones correspondientes
        self.btn_arriba.clicked.connect(self.mostrarInterfazFoto)
        self.btn_abajo.clicked.connect(self.mostrarInterfazManual)
        self.btn_mas_abajo.clicked.connect(self.capturaFrame2)

        self.layout.addWidget(self.btn_arriba)
        self.layout.addWidget(self.btn_abajo)
        self.layout.addWidget(self.btn_mas_abajo)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        self.setLayout(self.layout)


    def mostrarInterfazFoto(self):
        
        # Crear la interfaz cuando se presiona el botón "Arriba"
        self.interfaz_arriba = QWidget()
        layout_arriba = QVBoxLayout()

        self.depth_label = QLabel('Ingrese la profundidad:')
        self.depth_input = QLineEdit()
        layout_arriba.addWidget(self.depth_label)
        layout_arriba.addWidget(self.depth_input)

        self.base_line_label = QLabel('Ingrese Linea base:')
        self.base_line_input = QLineEdit()
        layout_arriba.addWidget(self.base_line_label)
        layout_arriba.addWidget(self.base_line_input)

        self.calc_button = QPushButton('Calcular Distancia Focal')
        self.calc_button.clicked.connect(self.calcular_distancia_focal)
        layout_arriba.addWidget(self.calc_button)

        self.result_label = QLabel('')
        layout_arriba.addWidget(self.result_label)

        self.interfaz_arriba.setLayout(layout_arriba)
        self.stacked_widget.addWidget(self.interfaz_arriba)
        self.stacked_widget.setCurrentWidget(self.interfaz_arriba)

    def mostrarInterfazManual(self):
        # Crear la interfaz cuando se presiona el botón "Abajo"
        self.interfaz_abajo = QWidget()
        layout_abajo = QVBoxLayout()

        self.res_x_label = QLabel('Resolución X:')
        self.res_x_input = QLineEdit()

        self.res_y_label = QLabel('Resolución Y:')
        self.res_y_input = QLineEdit()

        self.dist_focal_label = QLabel('Distancia Focal (mm):')
        self.dist_focal_input = QLineEdit()

        self.sensor_x_label = QLabel('Tamaño Sensor X (mm):')
        self.sensor_x_input = QLineEdit()

        self.sensor_y_label = QLabel('Tamaño Sensor Y (mm):')
        self.sensor_y_input = QLineEdit()

        self.base_line_label = QLabel('Ingrese Linea base:')
        self.base_line_input = QLineEdit()
        

        self.calc_button = QPushButton('Calcular Distancia Focal')
        self.calc_button.clicked.connect(self.calcular_distancia_focal_manual)

        self.result_label = QLabel('')
     


        layout_abajo.addWidget(self.res_x_label)
        layout_abajo.addWidget(self.res_x_input)
        layout_abajo.addWidget(self.res_y_label)
        layout_abajo.addWidget(self.res_y_input)
        layout_abajo.addWidget(self.dist_focal_label)
        layout_abajo.addWidget(self.dist_focal_input)
        layout_abajo.addWidget(self.sensor_x_label)
        layout_abajo.addWidget(self.sensor_x_input)
        layout_abajo.addWidget(self.sensor_y_label)
        layout_abajo.addWidget(self.sensor_y_input)
        layout_abajo.addWidget(self.base_line_label)
        layout_abajo.addWidget(self.base_line_input)
        layout_abajo.addWidget(self.calc_button)
        layout_abajo.addWidget(self.result_label)



        self.interfaz_abajo.setLayout(layout_abajo)
        self.stacked_widget.addWidget(self.interfaz_abajo)
        self.stacked_widget.setCurrentWidget(self.interfaz_abajo)

    def calcular_distancia_focal_manual(self):
        # Obtener los valores ingresados por el usuario
        res_x_input = self.res_x_input.text()
        res_y_input = self.res_y_input.text()
        dist_focal_input = self.dist_focal_input.text()
        sensor_x_input = self.sensor_x_input.text()
        sensor_y_input = self.sensor_y_input.text()
        baseline = self.base_line_input.text()

        # Convertir los valores a valores numéricos (aquí se puede hacer validación adicional)
        try:
            res_x = float(res_x_input)
            res_y = float(res_y_input)
            dist_focal = float(dist_focal_input)
            sensor_x = float(sensor_x_input)
            sensor_y = float(sensor_y_input)

            if res_x == 0 or res_y == 0 or dist_focal==0 or sensor_x==0 or sensor_y==0 or baseline==0 or res_x < 0 or res_y < 0 or dist_focal<0 or sensor_x<0 or sensor_y<0 or int(baseline)<0:
                self.result_label.setText('Ingrese números distintos a 0 y positivo')
                return

         
        except ValueError:
            self.result_label.setText('Ingrese números válidos')
            return

        
        dist_focal_manual = (res_x * dist_focal) / sensor_x



        with open('DB/Almacen/baseline.txt', 'w') as archivo:
            archivo.write(str(baseline))

        with open('DB/Almacen/distanciaFocal.txt', 'w') as archivo:
            archivo.write(str(dist_focal_manual))

        # Mostrar la distancia focal calculada
        self.result_label.setText(f'Distancia Focal Calculada: {dist_focal_manual} píxeles')
        self._write(f"Distancia focal (px): {dist_focal_manual}\n")

    def calcular_distancia_focal(self):
        # Obtener la profundidad ingresada por el usuario
        depth_input = self.depth_input.text()
        base_line_input = self.base_line_input.text()
        
        # Convertir la profundidad a un valor numérico (aquí se puede hacer validación adicional)
        try:
            depth = float(depth_input)
            baseline = float(base_line_input)

            if depth == 0 or baseline == 0 or depth < 0 or baseline < 0:
                self.result_label.setText('Ingrese números distintos a 0 y positivo')
                return
            
        except ValueError:
            self.result_label.setText('Ingrese un número válido para la profundidad')
            return

        self.calculos.calculos()

        # Calcular distancia focal
        focal_length_calculated = (depth * self.calculos.disparidad) / baseline


        with open('DB/Almacen/distanciaFocal.txt', 'w') as archivo:
            archivo.write(str(focal_length_calculated))

        with open("DB/Almacen/baseline.txt", 'w') as archivo:
            archivo.write(str(base_line_input))

        # Mostrar la distancia focal calculada
        self.result_label.setText(f'Distancia Focal Calculada: {focal_length_calculated} píxeles')
        self._write(f"Distancia focal (px): {focal_length_calculated}\n")


    def plot_boxes(self, results):
        xyxys = []
        confidences = []
        class_ids = []
        centers = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            xyxyss  = boxes.xyxy

            for xyxy in xyxyss:
               centro_x = (int(xyxy[0]) + int(xyxy[2]))/2
               centro_y = (int(xyxy[1]) + int(xyxy[3]))/2
               centers.append((centro_x, centro_y))

            #print(centers)
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)
        
        
        return results[0].plot(), xyxys, confidences, class_ids
    
    def capturaFrame2(self):
        self._write("Starting Camera\n")
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Cannot open camera"

        while True:  # Tomar dos fotos
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar la detección de objetos (utilizando tu modelo de detección)
            results = self.model(frame)
            frame_with_boxes, _, _, _ = self.plot_boxes(results)

            # Mostrar el fotograma con las cajas dibujadas
            cv2.imshow('Object Detection', frame_with_boxes)

            # Esperar la entrada del teclado
            key = cv2.waitKey(1)

            # Si se presiona la tecla 'c', guardar el fotograma y contar
            if key & 0xFF == ord('c'):
                cv2.imwrite(f'DB/fotograma_1.png', frame)
                self._write(f"Fotograma foto de cam 1 guardada como 'fotograma_1.png'\n")
         

            # Presionar 'q' para salir del bucle
            if key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyWindow('Object Detection')
                break

        self._write("Starting Camera\n")
        cap2 = cv2.VideoCapture(1)
        assert cap2.isOpened(), "Cannot open camera"

        while True:  # Tomar dos fotos
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # Realizar la detección de objetos (utilizando tu modelo de detección)
            results2 = self.model(frame2)
            frame_with_boxes2, _, _, _ = self.plot_boxes(results2)

            # Mostrar el fotograma con las cajas dibujadas
            cv2.imshow('Object Detection', frame_with_boxes2)

            # Esperar la entrada del teclado
            key = cv2.waitKey(1)

            # Si se presiona la tecla 'c', guardar el fotograma y contar
            if key & 0xFF == ord('c'):
                cv2.imwrite(f'DB/fotograma_2.png', frame2)
                self._write(f"Fotograma foto de cam 2 guardada como 'fotograma_2.png'\n")
         

            # Presionar 'q' para salir del bucle
            if key & 0xFF == ord('q'):
                cap2.release()
                cv2.destroyWindow('Object Detection')
                break

    def _write(self, content):
        try:
            with open("DB/Almacen/log.txt", 'a') as archivo:
                archivo.write(content)
            print(f"Se ha escrito correctamente en el log.")
        except Exception as e:
            print(f"Error al escribir en el log: {e}")



