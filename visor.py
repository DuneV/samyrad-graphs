import numpy as np
import open3d as o3d
import time

vectors = np.load("depth_vectors.npy")

def depth_to_pointcloud(depth_frame, fx=525.0, fy=525.0, cx=None, cy=None):
    """Convierte un depth frame a nube de puntos 3D"""
    h, w = depth_frame.shape
    cx = cx or w / 2
    cy = cy or h / 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filtrar puntos con profundidad 0 o inválida
    mask = z.flatten() > 0
    points = points[mask]

    return points


def get_colored_points(depth_frame):
    """Obtiene los puntos y sus colores basados en la profundidad"""
    points = depth_to_pointcloud(depth_frame)
    
    # Calcular colores (Rojo = Lejos, Azul = Cerca)
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
    
    colors = np.zeros((len(z_norm), 3))
    colors[:, 0] = z_norm        # Canal Rojo
    colors[:, 2] = 1 - z_norm    # Canal Azul
    
    return points, colors


def play_pointcloud_video(fps=30):
    total_frames = len(vectors)
    
    # Variables de control usando listas para poder modificarlas dentro de los callbacks
    state = {
        "frame_idx": 0,
        "is_playing": True,
        "last_time": time.time()
    }
    
    # 1. Inicializar el Visualizador
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud Video Player", width=1280, height=720)

    # Crear el objeto de nube de puntos inicial (Frame 0)
    pcd = o3d.geometry.PointCloud()
    points, colors = get_colored_points(vectors[0])
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # --- Función que se ejecuta en cada ciclo de renderizado ---
    def update_video(vis):
        current_time = time.time()
        # Controlar la velocidad de reproducción según los FPS deseados
        if current_time - state["last_time"] < (1.0 / fps):
            return False # No actualizar todavía

        if state["is_playing"]:
            state["frame_idx"] = (state["frame_idx"] + 1) % total_frames
            
            # Obtener datos del nuevo frame
            points, colors = get_colored_points(vectors[state["frame_idx"]])
            
            # Actualizar la geometría existente en Open3D
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Notificar al visualizador que los datos cambiaron
            vis.update_geometry(pcd)
            
            state["last_time"] = current_time
            
        return True # Indica que se debe redibujar la pantalla

    # --- Callbacks de teclado para control del video ---
    def toggle_play(vis):
        state["is_playing"] = not state["is_playing"]
        status = "REPRODUCIENDO" if state["is_playing"] else "PAUSADO"
        print(f"[{status}] - Frame actual: {state["frame_idx"]}")

    def reset_video(vis):
        state["frame_idx"] = 0
        print("Video reiniciado al inicio.")

    # Registramos la función de actualización automática
    vis.register_animation_callback(update_video)
    
    # Teclas de control adicionales:
    # Espacio (32) para Pausar/Reproducir
    vis.register_key_callback(32, toggle_play)
    # R (82) para Reiniciar el video
    vis.register_key_callback(82, reset_video)

    print("\n" + "="*50)
    print(" REPRODUCTOR DE NUBE DE PUNTOS")
    print("="*50)
    print(" - [Espacio]: Pausar / Reproducir")
    print(" - [R]: Reiniciar video")
    print(" - [Q]: Salir")
    print(" " + "="*50 + "\n")

    # Iniciar bucle de renderizado
    vis.run()
    vis.destroy_window()

# Ejecutar el reproductor de video a 30 FPS (puedes ajustar este número según tu PC)
play_pointcloud_video(fps=30)