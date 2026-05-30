import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLOE
import json

class YoloSegmentationNode(Node):
    def __init__(self):
        super().__init__('yolo_segmentation_node')
        
        # Declarar parámetros
        self.declare_parameter('model_path', 'yoloe-11l-seg-pf.pt')  # Default Values, they can be change in launch file
        self.declare_parameter('confidence_threshold', 0.5)     # Default Values, they can be change in launch file
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/yolo/segmented_image')
        self.declare_parameter('results_topic', '/yolo/detection_results')
        
        # Obtener parámetros
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        results_topic = self.get_parameter('results_topic').get_parameter_value().string_value
        
        # Inicializar YOLO
        try:
            self.model = YOLOE(model_path)
            self.get_logger().info(f'YOLO-E model loaded: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO-E model: {e}')
            return
        
        # Bridge para conversión de imágenes
        self.bridge = CvBridge()
        
        # Publisher y Subscriber
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10
        )
        
        self.image_pub = self.create_publisher(
            Image,
            output_topic,
            10
        )
        
        # Publisher para máscaras (opcional)
        self.mask_pub = self.create_publisher(
            Image,
            '/yolo/segmentation_mask',
            10
        )
        

        self.results_pub = self.create_publisher(
            String,  # Usamos String para JSON
            results_topic,
            10
        )

        self.stable_memory = {}
        self.confirmation_threshold = 45

        self.get_logger().info('YOLO Segmentation Node initialized')
        
    def image_callback(self, msg):
        try:
            # Convertir imagen ROS a OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # Realizar inferencia
            results = self.model(cv_image, conf=self.confidence_threshold)
            
            # Procesar resultados
            annotated_image, combined_mask, detection_results = self.process_results(cv_image, results)
            
            # --- FILTRO TEMPORAL DE DETECCIONES ---
            current_frame_objects = [det['class_name'] for det in detection_results]

            # Actualizar contadores
            for obj in current_frame_objects:
                self.stable_memory[obj] = self.stable_memory.get(obj, 0) + 1

            # Resetear objetos que desaparecieron
            for obj in list(self.stable_memory.keys()):
                if obj not in current_frame_objects:
                    self.stable_memory[obj] = 0

            # Mantener solo objetos confirmados
            confirmed_objects = {
                obj for obj, count in self.stable_memory.items()
                if count >= self.confirmation_threshold
            }

            # Filtrar detecciones
            filtered_results = [
                det for det in detection_results
                if det['class_name'] in confirmed_objects
            ]

            # Sobrescribir antes de publicar
            detection_results = filtered_results

            # Publicar imagen segmentada
            if annotated_image is not None:
                seg_msg = self.bridge.cv2_to_imgmsg(annotated_image, "rgb8")
                seg_msg.header = msg.header
                self.image_pub.publish(seg_msg)
            
            # Publicar máscara combinada
            if combined_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, "mono8")
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)

            self.get_logger().info(f"Procesando imagen. Detecciones encontradas: {len(detection_results)}")
            if detection_results:
                results_msg = String()
                results_data = {
                    'header': {
                        'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                        'frame_id': msg.header.frame_id
                    },
                    'detections': detection_results,
                    'total_detections': len(detection_results)
                }
                results_msg.data = json.dumps(results_data, indent=2)
                self.results_pub.publish(results_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def process_results(self, image, results):
        """Procesar resultados de YOLO y crear visualización"""
        annotated_image = image.copy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        detection_results = []  # Lista para almacenar resultados estructurados
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Procesar cada detección
                for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
                    if conf >= self.confidence_threshold:
                        # Redimensionar máscara al tamaño de la imagen
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        
                        # Agregar a la máscara combinada
                        combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                        
                        # Crear overlay colorizado
                        color = self.get_class_color(int(cls))
                        colored_mask = np.zeros_like(image)
                        colored_mask[mask_binary > 0] = color
                        
                        # Aplicar máscara con transparencia
                        alpha = 0
                        annotated_image = cv2.addWeighted(annotated_image, 1-alpha, colored_mask, alpha, 0)
                        
                        # Dibujar bounding box y etiqueta
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.model.names[int(cls)]
                        
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Etiqueta con fondo
                        label = f'{class_name}: {conf:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_image, (x1, y1-25), (x1+label_size[0], y1), color, -1)
                        cv2.putText(annotated_image, label, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Crear resultado estructurado para esta detección
                        # Crear resultado estructurado simplificado
                        detection_result = {
                            'class_name': class_name,
                            'confidence': float(conf),
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }

                        detection_results.append(detection_result)
        
        return annotated_image, combined_mask, detection_results
    
    def compress_mask(self, mask):
        """Comprimir máscara para transmisión eficiente"""
        # Encuentra contornos de la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convierte contornos a lista de puntos
        compressed_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filtrar contornos muy pequeños
                # Simplificar contorno
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convertir a lista de puntos
                points = []
                for point in approx:
                    points.append([int(point[0][0]), int(point[0][1])])
                
                compressed_contours.append(points)
        
        return compressed_contours
    
    def get_class_color(self, class_id):
        """Obtener color único para cada clase"""
        colors = [
            (0, 114, 178),   # Azul profundo
            (230, 159, 0),   # Naranja fuerte
            (86, 180, 233),  # Azul cielo
            (204, 121, 167), # Magenta rosado
            (0, 158, 115),   # Verde-azulado
            (213, 94, 0),    # Rojo anaranjado 
            (240, 228, 66),  # Amarillo brillante
        ]
        return colors[class_id % len(colors)]

def main(args=None):
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()