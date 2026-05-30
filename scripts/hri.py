import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import select

class UserInterfaceNode(Node):
    def __init__(self):
        super().__init__('hri')
        
        self.goal_pub = self.create_publisher(String, '/user_input', 100)
        
        # Subscriber para confirmaciones
        self.feedback_sub = self.create_subscription(
            String,
            '/neopath/goal_results',
            self.feedback_callback,
            100
        )
        
        self.waiting_for_feedback = False

        print("\n")
        print("¡Hola! Yo soy Aura, tu robot asistente.")
        print("¿Que puedo hacer por ti el dia de hoy?\n", end="", flush=True)
        
        # Timer no bloqueante para leer input
        self.create_timer(0.1, self.check_input)
        
    def check_input(self):
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line:
                msg = String()
                msg.data = line
                self.goal_pub.publish(msg)
                self.waiting_for_feedback = True
    
    def feedback_callback(self, msg):
        if not msg.data.startswith("search_for("):
            self.waiting_for_feedback = False
            print("\n¿Qué más puedo hacer por ti?: ", end='', flush=True)

def main():
    rclpy.init()
    node = UserInterfaceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()