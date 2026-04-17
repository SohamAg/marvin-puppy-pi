#!/usr/bin/env python3

import time
import rclpy
import socket
import threading

from actions.actions import PuppyActions
from inference.inference import get_command

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

wakeWord = 1 # Set to 0 if not using wakeword detection
wake_event = threading.Event()

def start_server(stop_event):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    server.settimeout(1.0)
    print("Server is listening for connections...")

    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()
            print(f"New connection added at {addr}")
        except socket.timeout:
            continue
    server.close()

def handle_client(conn, addr):
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"Data received: {data} ")
            if data == b"Marv":
                wake_event.set()
    except Exception:
        pass

def main(args=None):
    stop_signal = threading.Event()
    server_thread = threading.Thread(target=start_server, args=(stop_signal,), daemon=True)
    server_thread.start()
    if wakeWord:
        wake_event.wait()

    rclpy.init(args=args)
    controller = PuppyActions()

    try:
        time.sleep(1.0)
        controller.stand()
        time.sleep(1.0)

        while True:
            command = get_command()

            if command == 'stand':
                controller.stand()

            elif command == 'walk':
                controller.walk_forward(speed=6.0, gait_type='Amble')

            elif command == 'left':
                controller.turn_left()

            elif command == 'right':
                controller.turn_right()

            elif command == 'stop':
                controller.stop()

            elif command == 'quit':
                controller.stop()
                break

            else:
                print("Unknown command.")

    except KeyboardInterrupt:
        controller.stop()

    finally:
        stop_signal.set()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
