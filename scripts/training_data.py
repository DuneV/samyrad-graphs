from typing import List, Dict

def get_movement_scenarios() -> List[Dict]:
    """Escenarios de movimiento de objetos"""
    return [
        {
            'goal': 'Move the cup to the table',
            'target_objects': ['cup', 'table'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['cup', 'table', 'chair', 'bowl'],
            'object_positions': {
                'cup': (200, 250, 1.1),
                'table': (350, 300, 1.4),
                'chair': (280, 350, 1.5),
                'bowl': (420, 280, 1.3)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'cup'): 0.25,
                ('cup', 'move_to', 'table'): 0.3,
                ('robot_right_hand', 'press', 'table'): 1.8,
            }
        },
        {
            'goal': 'Move the book to the shelf',
            'target_objects': ['book', 'shelf'],
            'required_actions': ['move_to'],
            'detected_objects': ['book', 'shelf', 'table', 'chair'],
            'object_positions': {
                'book': (280, 290, 1.1),
                'shelf': (450, 320, 1.6),
                'table': (300, 330, 1.4),
                'chair': (250, 360, 1.5)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'book'): 0.2,
                ('book', 'move_to', 'shelf'): 0.25,
                ('chair', 'move_to', 'book'): 2.0,
                ('robot_camera', 'inspect', 'book'): 1.5
            }
        },
        {
            'goal': 'Place the plate on the dining table',
            'target_objects': ['plate', 'dining_table'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['plate', 'dining_table', 'fork', 'knife'],
            'object_positions': {
                'plate': (200, 250, 1.1),
                'dining table': (350, 320, 1.5),
                'fork': (180, 270, 1.05),
                'knife': (220, 260, 1.08)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'plate'): 0.2,
                ('plate', 'move_to', 'dining table'): 0.3,
            }
        },
        {
            'goal': 'Move the laptop to the desk',
            'target_objects': ['laptop', 'desk'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['laptop', 'desk', 'mouse', 'keyboard'],
            'object_positions': {
                'laptop': (210, 260, 1.2),
                'desk': (400, 310, 1.4),
                'mouse': (230, 290, 1.1),
                'keyboard': (260, 300, 1.2)
            },
            'edge_labels': {
                ('robot_hands', 'grasp', 'laptop'): 0.25,
                ('laptop', 'move_to', 'desk'): 0.32,
                ('robot_right_hand', 'grasp', 'laptop'): 0.6
            }
        },
        {
            'goal': 'Move the bottle to the counter',
            'target_objects': ['bottle', 'counter'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['bottle', 'counter', 'sink', 'plate'],
            'object_positions': {
                'bottle': (180, 200, 1.1),
                'counter': (350, 220, 1.3),
                'sink': (300, 240, 1.4),
                'plate': (260, 260, 1.2),
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'bottle'): 0.2,
                ('bottle', 'move_to', 'counter'): 0.35,
            }
        },
        {
            'goal': 'Move the phone to the office desk',
            'target_objects': ['phone', 'office desk'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['phone', 'office desk', 'lamp'],
            'object_positions': {
                'phone': (140, 200, 1.0),
                'charger_stand': (380, 260, 1.3),
                'lamp': (260, 280, 1.5),
            },
            'edge_labels': {
                ('robot_left_hand', 'grasp', 'phone'): 0.18,
                ('phone', 'move_to', 'office_desk'): 0.29,
            }
        },
        {
            'goal': 'Move the toy to the cabinet',
            'target_objects': ['toy', 'cabinet'],
            'required_actions': ['move_to'],
            'detected_objects': ['toy', 'cabinet', 'ball'],
            'object_positions': {
                'toy': (150, 210, 1.1),
                'toy_box': (400, 350, 1.4),
                'ball': (180, 230, 1.0)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'toy'): 0.1,
                ('toy', 'move_to', 'cabinet'): 0.33,
            }
        },
        {
            'goal': 'Move the pillow to the bed',
            'target_objects': ['pillow', 'bed'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['pillow', 'bed', 'blanket'],
            'object_positions': {
                'pillow': (220, 300, 1.2),
                'bed': (450, 350, 1.5),
                'blanket': (260, 310, 1.4)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'pillow'): 0.27,
                ('pillow', 'move_to', 'bed'): 0.36,
                ('robot_right_hand', 'push', 'pillow'): 1.6,
            }
        },
        {
            'goal': 'Move the apple to the basket',
            'target_objects': ['apple', 'basket'],
            'required_actions': ['grasp', 'move_to'],
            'detected_objects': ['apple', 'basket', 'banana'],
            'object_positions': {
                'apple': (200, 250, 1.0),
                'basket': (380, 300, 1.4),
                'banana': (230, 260, 1.1),
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'apple'): 0.21,
                ('apple', 'move_to', 'basket'): 0.34,
            }
        },
        {
            'goal': 'Remove the keyboard from the desk',
            'target_objects': ['keyboard', 'desk'],
            'required_actions': ['grasp', 'remove_from'],
            'detected_objects': ['keyboard', 'desk', 'mouse', 'mousepad'],
            'object_positions': {
                'keyboard': (260, 300, 1.2),
                'desk': (400, 310, 1.4),
                'mouse': (230, 290, 1.1),
                'mousepad': (250, 295, 1.15)
            },
            'edge_labels': {
                ('robot_left_hand', 'grasp', 'keyboard'): 0.25,
                ('keyboard', 'remove_from', 'desk'): 0.3,
                # penalizar moverlo hacia el desk
                ('keyboard', 'move_to', 'desk'): 2.0
            }
        },
        {
            'goal': 'Remove the knife from the table',
            'target_objects': ['knife', 'table'],
            'required_actions': ['grasp', 'remove_from'],
            'detected_objects': ['knife', 'table', 'plate'],
            'object_positions': {
                'knife': (240, 260, 1.05),
                'table': (350, 300, 1.4),
                'plate': (260, 280, 1.1)
            },
            'edge_labels': {
                ('robot_left_hand', 'grasp', 'knife'): 0.22,
                ('knife', 'remove_from', 'table'): 0.25,
                ('knife', 'move_to', 'table'): 2.5
            }
        },
        {
            'goal': 'Remove the knife from the table',
            'target_objects': ['knife', 'table'],
            'required_actions': ['grasp', 'remove_from'],
            'detected_objects': ['knife', 'table', 'plate'],
            'object_positions': {
                'knife': (240, 260, 1.05),
                'table': (350, 300, 1.4),
                'plate': (260, 280, 1.1)
            },
            'edge_labels': {
                ('robot_left_hand', 'grasp', 'knife'): 0.22,
                ('knife', 'remove_from', 'table'): 0.25,
                ('knife', 'move_to', 'table'): 2.5
            }
        },
        {
            'goal': 'Remove the cup from the table',
            'target_objects': ['cup', 'table'],
            'required_actions': ['grasp', 'remove_from'],
            'detected_objects': ['cup', 'table', 'plate'],
            'object_positions': {
                'cup': (200, 250, 1.1),
                'table': (350, 300, 1.4),
                'plate': (230, 270, 1.15)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'cup'): 0.2,
                ('cup', 'remove_from', 'table'): 0.28,
                ('cup', 'move_to', 'table'): 1.9
            }
        },
        # Agregar más ejemplos aquí...
    ]

def get_manipulation_scenarios() -> List[Dict]:
    """Escenarios de manipulación (abrir, cerrar, etc.)"""
    return [
        {
            'goal': 'Open the cabinet to access utensils',
            'target_objects': ['cabinet'],
            'required_actions': ['open'],
            'detected_objects': ['cabinet', 'table', 'drawer', 'chair'],
            'object_positions': {
                'cabinet': (320, 300, 1.5),
                'table': (350, 320, 1.6),
                'drawer': (300, 280, 1.3),
                'chair': (280, 350, 1.7)
            },
            'edge_labels': {
                ('robot_right_hand', 'open', 'cabinet'): 0.3,
                ('robot_right_hand', 'grasp', 'cabinet'):1.3,
            }
        },
        {
            'goal': 'Close the drawer after use',
            'target_objects': ['drawer'],
            'required_actions': ['close'],
            'detected_objects': ['drawer', 'cabinet', 'table'],
            'object_positions': {
                'drawer': (300, 280, 1.3),
                'cabinet': (320, 300, 1.5),
                'table': (350, 320, 1.6)
            },
            'edge_labels': {
                ('robot_right_hand', 'close', 'drawer'): 0.2,
                ('robot_right_hand', 'grasp', 'drawer'):1.4,
                ('robot_right_hand', 'push', 'drawer'): 0.8
            }
        },
        {
            'goal': 'Open the fridge door',
            'target_objects': ['fridge'],
            'required_actions': ['open'],
            'detected_objects': ['fridge', 'counter', 'microwave', 'chair'],
            'object_positions': {
                'fridge': (400, 300, 1.7),
                'counter': (350, 330, 1.6),
                'microwave': (320, 290, 1.4),
                'chair': (280, 350, 1.5)
            },
            'edge_labels': {
                ('robot_left_hand', 'open', 'fridge'): 0.35,
                # caminos falsos muy costosos
                ('robot_right_hand', 'pull', 'fridge'): 1.8,
                ('robot_camera', 'inspect', 'microwave'): 1.9,
            }
        },

        {
            'goal': 'Push the door to close it',
            'target_objects': ['door'],
            'required_actions': ['push'],
            'detected_objects': ['door', 'mirror', 'table'],
            'object_positions': {
                'door': (300, 250, 1.9),
                'mirror': (260, 260, 1.8),
                'table': (350, 300, 1.6)
            },
            'edge_labels': {
                ('robot_right_hand', 'push', 'door'): 0.28,
                ('robot_right_hand', 'pull', 'door'): 1.95,
            }
        },

        {
            'goal': 'Rotate the knob to turn on the stove',
            'target_objects': ['stove_knob'],
            'required_actions': ['rotate'],
            'detected_objects': ['stove_knob', 'pan', 'bottle'],
            'object_positions': {
                'stove_knob': (250, 270, 1.1),
                'pan': (200, 260, 1.0),
                'bottle': (180, 240, 1.1)
            },
            'edge_labels': {
                ('robot_right_hand', 'rotate', 'stove_knob'): 0.22,
                ('robot_right_hand', 'grasp', 'stove_knob'): 1.0,
                ('robot_right_hand', 'push', 'stove_knob'): 1.7,
            }
        },

        {
            'goal': 'Press the button to start the tv',
            'target_objects': ['remote'],
            'required_actions': ['press'],
            'detected_objects': ['button', 'panel', 'lever'],
            'object_positions': {
                'button': (300, 260, 1.3),
                'panel': (320, 300, 1.4),
                'lever': (280, 250, 1.2)
            },
            'edge_labels': {
                ('robot_left_hand', 'press', 'button'): 0.18,
                ('robot_left_hand', 'grasp', 'button'): 0.8,
                ('robot_left_hand', 'rotate', 'button'): 1.8
            }
        },

        {
            'goal': 'Pull the drawer to open it',
            'target_objects': ['drawer'],
            'required_actions': ['pull'],
            'detected_objects': ['drawer', 'cabinet', 'lamp'],
            'object_positions': {
                'drawer': (320, 300, 1.3),
                'cabinet': (300, 280, 1.4),
                'lamp': (350, 330, 1.5)
            },
            'edge_labels': {
                ('robot_right_hand', 'pull', 'drawer'): 0.24,
                ('robot_right_hand', 'push', 'drawer'): 2.0,
                ('lamp', 'rotate', 'cabinet'): 1.6,
            }
        },

        {
            'goal': 'Twist the cap to open the bottle',
            'target_objects': ['bottle_cap', 'bottle'],
            'required_actions': ['rotate'],
            'detected_objects': ['bottle', 'bottle_cap', 'cup'],
            'object_positions': {
                'bottle': (210, 220, 1.1),
                'bottle_cap': (210, 215, 1.15),
                'cup': (300, 260, 1.4)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'bottle'): 0.26,
                ('robot_left_hand', 'rotate', 'bottle_cap'): 0.1,
            }
        },

        {
            'goal': 'Slide the window to open it',
            'target_objects': ['window'],
            'required_actions': ['push'],
            'detected_objects': ['window', 'curtain', 'desk'],
            'object_positions': {
                'window': (450, 320, 1.8),
                'curtain': (430, 330, 1.7),
                'desk': (380, 310, 1.5)
            },
            'edge_labels': {
                ('robot_right_hand', 'push', 'window'): 0.30,
            }
        },
        # Más ejemplos...
    ]

def get_cooking_scenarios() -> List[Dict]:
    """Escenarios de cocina"""
    return [
        {
            'goal': 'Cook an egg by roasting it on the pan',
            'target_objects': ['egg', 'pan', 'spatula', 'stove'],
            'required_actions': ['grasp', 'move', 'roast', 'flip'],
            'detected_objects': ['egg', 'pan', 'spatula', 'stove', 'bowl'],
            'object_positions': {
                'egg': (150, 220, 1.1),
                'pan': (200, 240, 1.2),
                'spatula': (280, 200, 0.9),
                'stove': (180, 260, 1.3),
                'bowl': (400, 280, 1.4)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'egg'): 0.2,
                ('robot_left_hand', 'grasp', 'pan'): 0.25,
                ('pan', 'roast', 'egg'): 0.3,
                ('robot_left_hand', 'grasp', 'spatula'): 0.4,
                ('spatula', 'flip', 'egg'): 0.35,
                ('robot_right_hand', 'push', 'egg'): 0.9,
                ('robot_left_hand', 'pull', 'pan'): 1.1,
                ('robot_left_hand', 'rotate', 'spatula'): 1.6,
            }
        },
        {
            'goal': 'Pour water from bottle into cup',
            'target_objects': ['bottle', 'cup'],
            'required_actions': ['grasp', 'pour'],
            'detected_objects': ['bottle', 'cup', 'table'],
            'object_positions': {
                'bottle': (180, 230, 1.1),
                'cup': (350, 270, 1.3),
                'table': (280, 300, 1.4)
            },
            'edge_labels': {
                ('robot_right_hand', 'grasp', 'bottle'): 0.2,
                ('bottle', 'pour', 'cup'):0.25,
                ('robot_right_hand', 'push', 'bottle'): 1.8
            }
        
        },
        # Más ejemplos...
    ]

def get_difficult_scenarios() -> List[Dict]:
    """Casos difíciles: objetos no detectados, muy lejanos, etc."""
    return [
        {
            'goal': 'Move the vase to the shelf',
            'target_objects': ['vase', 'shelf'],
            'required_actions': ['grasp', 'move'],
            'detected_objects': ['shelf', 'book', 'clock'],  # vase NO detectado
            'object_positions': {
                'shelf': (400, 320, 1.6),
                'book': (280, 290, 1.1),
                'clock': (420, 300, 1.55)
            },
            'edge_labels': {
                ('robot_camera', 'search_for', 'vase'): 0.4,
                ('robot_gripper', 'grasp', 'vase'): 0.7,
                ('vase', 'move_to', 'shelf'): 0.3,
            }
        },
        {
            'goal': 'Move the remote to the sofa',
            'target_objects': ['remote control', 'sofa'],
            'required_actions': ['grasp', 'move'],
            'detected_objects': ['remote control', 'sofa', 'tv'],
            'object_positions': {
                'remote control': (150, 200, 1.0),
                'sofa': (500, 400, 1.8),  # MUY LEJOS
                'tv': (300, 250, 1.4)
            },
            'edge_labels': {
                ('robot_gripper', 'grasp', 'remote control'): 0.3,
                ('remote control', 'move_to', 'sofa'): 0.5,
            }
        },
        
        # Más casos difíciles...
    ]

def get_all_training_data() -> List[Dict]:
    """
    Función principal que retorna TODOS los datos de entrenamiento.
    Esta es la función que llamarás desde train_gnn.py
    """
    all_data = []
    
    all_data.extend(get_movement_scenarios())
    all_data.extend(get_manipulation_scenarios())
    all_data.extend(get_cooking_scenarios())
    all_data.extend(get_difficult_scenarios())
    
    print(f" Total de ejemplos de entrenamiento: {len(all_data)}")
    print(f"   - Movimiento: {len(get_movement_scenarios())}")
    print(f"   - Manipulación: {len(get_manipulation_scenarios())}")
    print(f"   - Cocina: {len(get_cooking_scenarios())}")
    print(f"   - Casos difíciles: {len(get_difficult_scenarios())}")
    
    return all_data