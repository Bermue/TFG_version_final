
#Autor: Carlos Bermúdez Expósito

# CoordenadasDobleCam_V3
# Uso de un modelo entrenado para reconocimiento de objetos y calculo de profundidad mediante 2 cámaras.
# Posicionamiento de los objetos, por coordenada y pk, en un mapa y cálculo de la velocidad media.


import csv
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import folium
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
from ultralytics.utils.plotting import Annotator



class ObjectDetection:
    def __init__(self, capture_index, capture_index2, csv_path, map_html_path):
        #index de la cámara1
        self.capture_index = capture_index

        #index de la cámara2
        self.capture_index2 = capture_index2

        #elegir si usar nucleos cuda o cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #mostrar el dispositivo que se está usando
        self._write(f"Using Device: {self.device}" )

        #cargar el modelo
        self.model = self.load_model()

        #cargar el csv con las coordenadas
        self.csv_path = csv_path
        self.load_csv()

        #cargar el mapa
        self.map_html_path = map_html_path

        #inicializar la lista de coordenadas visitadas
        self.coordenadas_visitadas = []

        #inicializar el tiempo para las velocidades
        self.last_time = datetime.now()

        #inicializar la lista de duplicados
        self.dupli = []

        #inicializar las coordenadas del GPS
        self.gps = [40.537027848989936, -3.8862253265456395]

        #inicializar la profundidad
        self.depth_left = 0

        #inicializar la corrección del PK
        self.correccion_PK = 0

        self.baseline = 0

        self.focal_length = 0
    
    #Carga el modelo entrenado de YOLO
    def load_model(self):
        model = YOLO("weights/best90.pt") 
        model.fuse()
        return model
    
    #Carga el csv y guarda en self.object_data[] los datos en diccionarios
    def load_csv(self):
        self.object_data = []
        with open(self.csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # saltar la primera linea, que es la de las cabeceras
            for row in csv_reader:
                self.object_data.append({'ID': int(row[2]), 'PK': int(row[3]), 'Longitude': float(row[0]), 'Latitude': float(row[1])})

    #Predice los objetos en el frame
    def predict(self, frame):
        results = self.model(frame)
        return results

    
    #Encuentra el objeto más cercano al gps del tren en el csv
    def find_nearest_object_in_csv(self, object_id):
        min_distance = float('inf')
        nearest_obj = None

        for obj in self.object_data:
            if object_id in self.dupli:
                self._write(f"Duplicado del: {object_id}")
                return None

            if obj['ID'] != object_id:
                continue

            obj_lat, obj_lon = obj['Longitude'], obj['Latitude']

            #print(f"las coordenas visitadas son: {self.coordenadas_visitadas}")


            distance = self.calculate_distance(self.gps[0], self.gps[1], obj_lat, obj_lon)
            #print(f"Distance: {distance} m de {obj['ID']}")

            if distance < min_distance:
                min_distance = distance
                nearest_obj = obj

        return nearest_obj
    
    #Calcula el PK del tren desde el de la señal
    def calculate_ajuste_PK(self, object, distancia_al_obj):
        
        obj_PK = object['PK']
        #Estos son los metros que hay, en teoría, de la señal al tren. Expresado en punto kilométrico
        ajuste_PK  = int(obj_PK) - distancia_al_obj 

        return ajuste_PK

   #Saca el centro de los objetos detectados en el eje X 
    def centers(self, results):

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
        
        

        return class_ids, centers
    

    #Calcula la profundidad de los objetos detectados en metros
    def calcular_profundidad(self,left_bbox, right_bbox, focal_length, baseline):

        centros_left = left_bbox[0]
        centro_x_left = centros_left[0]

        centros_right = right_bbox[0]
        centro_x_right = centros_right[0]


        # Calcular la disparidad (diferencia horizontal entre los centros)
        disparity = abs(centro_x_left - centro_x_right)

        # Calcular la profundidad en metros utilizando la fórmula de triangulación estéreo
        depth = (focal_length * baseline) / disparity
        return depth
    
    
    #Calcula la distancia entre coordenadas en metros
    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        radio_tierra = 6371.0

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distancia = radio_tierra * c

        return distancia
    
    #Actualiza el mapa con las coordenadas de los objetos
    def update_map(self, lat, lon, object_id):
        if self.last_time:
            current_time = datetime.now()
            time_diff = (current_time - self.last_time).seconds
            speed = self.calculate_speed(lat, lon, time_diff)
            if speed != 0:
                self._write(f"Velocidad: {speed} km/h")
            

        if (lat, lon) in self.coordenadas_visitadas:
            return
        self.coordenadas_visitadas.append((lat, lon))
        self.last_time = datetime.now()

        # Check if the map is already created, if not, create a new map
        if not hasattr(self, 'mapa'):
            self.mapa = folium.Map(location=[lat, lon], zoom_start=15)

        marker_color = 'green' if len(self.coordenadas_visitadas) == 1 else 'blue'

        folium.Marker([lat, lon], popup=f'Object ID: {object_id}', icon=folium.Icon(color=marker_color)).add_to(self.mapa)
        folium.PolyLine(locations=self.coordenadas_visitadas, color='blue', popup=self.coordenadas_visitadas).add_to(self.mapa)

        self.mapa.save(self.map_html_path)
    
    #Calcula la velocidad media
    def calculate_speed(self, lat, lon, time_diff):
        prev_lat, prev_lon = self.coordenadas_visitadas[-1] if self.coordenadas_visitadas else (lat, lon)
        distancia = self.calculate_distance(prev_lat, prev_lon, lat, lon)

        if time_diff != 0 and distancia != 0:
            velocidad = distancia / (time_diff / 3600)
            return velocidad
        else:
            return 0

    #Escribe el log    
    def _write(self, mensaje):
        print(mensaje)
        try:
            with open("DB/Almacen/log.txt", 'a') as archivo:
                archivo.write(f"{mensaje}\n")
            print(f"Se ha escrito correctamente en el log.")
        except Exception as e:
            print(f"Error al escribir en el log: {e}")

            
    #Dibuja los cuadrados de los objetos reconocidos con los datos de profundidad y PK
    def plot_boxes(self, frame, results):
        boxes = results[0].boxes.xyxy.cpu()
        annotator = Annotator(frame, line_width=3)
        for xyxy in boxes:
    
            annotator.box_label([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])], label=(f"{str(format(self.depth_left, '.2f'))}m, PK:{str(format(self.correccion_PK, '.2f'))}"), color=(0, 0, 255))
            self._write(f"Profundidad: {str(format(self.depth_left, '.2f'))}m, PK: {str(format(self.correccion_PK, '.2f'))}")

        return annotator.result()
    
    def read_distancia_focal(self):
        with open("DB/Almacen/distanciaFocal.txt", 'r') as file:
            try:
                # Lee el contenido del archivo y convierte a float
                float_number = float(file.read())
                return float_number
            except ValueError:
                print("Error: No se pudo convertir el contenido a un número float.")

    def read_baseline(self):
        try:
            with open("DB/Almacen/baseline.txt", 'r') as archivo:
                lineas = archivo.readlines()
                if lineas:
                    # Convertir el último número a flotante
                    ultimo_numero = float(lineas[-1].strip()) 
                    return ultimo_numero
                else:
                    return None
        except (FileNotFoundError, ValueError):
            return None
    

    #metodo que se llama al instanciar el objeto
    print("Loaded Model")
    def __call__(self):
   

        #inicializa la captura de las 2 cámaras
        print("Starting Camera")
        cap = cv2.VideoCapture(self.capture_index)
        cap2 = cv2.VideoCapture(self.capture_index2)
        assert cap.isOpened(), "Cannot open camera"
        assert cap2.isOpened(), "Cannot open camera"

        #inicializa la focal length y la baseline
        self.focal_length = self.read_distancia_focal()
        self.baseline = self.read_baseline()
        a = 0

        #bucle que lee los frames de las 2 cámaras
        while True:
            
            if a < 2:
                self.update_map(40.519105581236595,-3.8801884774552593,14)
                self.update_map(40.52425994149647,-3.8829779748557103,13)
                self.update_map(40.52303663236828,-3.8822913293417534,15)
                a = 3
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            if not ret:
                break
            if not ret2:
                break
            
            # Realizar la prediccion de cada frame sobre la detección de objetos
            results = self.predict(frame)
            results2 = self.predict(frame2)
           

            
            # Obtiene los centros de los objetos de la predicción
            class_ids, centers = self.centers(results) 
            class_ids2, centers2 = self.centers(results2)

            # Si las 2 cámaras detectan algún objeto y es el mismo, calcula el que está mas cerca del tren
            # actualiza el mapa con las coordenadas del objeto
            # calcula la profundidad del objeto
            # calcula el ajuste del PK y muestra las cajas con la predicción con los datos de profundidad y PK
            if len(centers) != 0  and len(centers2) != 0 :

        
                ids = class_ids[0][0]

     
                ids2 = class_ids2[0][0]
                if ids == ids2:
                    object_info = self.find_nearest_object_in_csv(ids)

                    if object_info:
                        self._write(f"Object ID: {ids}, Longitude: {object_info['Longitude']}, Latitude: {object_info['Latitude']}")
                        self.update_map(object_info['Longitude'], object_info['Latitude'], ids)
                        self.gps = object_info['Longitude'], object_info['Latitude']
                        self._write(f"el gps ahora es: {self.gps}")
                    else:
                        self._write(f"Object ID: {ids}, Not found in CSV")

                    self.depth_left = self.calcular_profundidad(centers, centers2, self.focal_length,self.baseline)
                    print(f"Profundidad = {self.focal_length} * {self.baseline} /disparity")
                    self._write(f"Profundidad estimada para id{ids}: {self.depth_left} metros")
                    self.correccion_PK = self.calculate_ajuste_PK(object_info, self.depth_left)

            annotator = self.plot_boxes(frame, results)
            annotator2 = self.plot_boxes(frame2, results2)


            key = cv2.waitKey(1)

            
            # Presionar 'q' para salir del bucle
            cv2.imshow('Object Detection', annotator)

            cv2.imshow('Object Detection2', annotator2)
            if key & 0xFF == ord('q'):
                break

            
            
     

        cap.release()
        cap2.release()
        cv2.destroyAllWindows()



def main():
    obj_detection = ObjectDetection(capture_index=0, capture_index2=1,csv_path='DB/Almacen/coordenadas.csv', map_html_path='DB/Almacen/mapa_objetos.html')
    obj_detection()

if __name__ == '__main__':
    main()

        
