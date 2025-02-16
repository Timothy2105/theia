from time import sleep
from socket import socket
from threading import Thread
from typing import Optional
from queue import Queue
import struct

q = Queue()

def pack_msg(msg: tuple[float, float, float]) -> bytes:
    # Expects (x, y, state) and packs into network byte order
    return struct.pack("!fff", *msg)

def create_server(port: int):
    """creates a listener socket"""
    lsock = socket()
    try:
        lsock.bind(("0.0.0.0", port))
        lsock.listen()
        return lsock
    except:
        print(f"failed to bind to addr 0.0.0.0:{port}")
        return None

def accept_socket(sock: socket) -> Optional[tuple[socket, tuple[str, str]]]:
    try:
        client_sock, client_addr = sock.accept()
        print(f"accepted connection from {client_addr}")
        return client_sock, client_addr
    except:
        print("failed to accept incoming connection")
        return None

def start(port = 2025):
    server_socket = create_server(port)
    if server_socket is None:
        return
    print(f"listening for connections on: 0.0.0.0:{port}")
    
    while True:
        client = accept_socket(server_socket)
        if client is None:
            return
        client_socket, client_addr = client
        print(f"client connected: {client_addr}")
        
        try:
            while item := q.get(block=True):
                # item should be (x, y, state)
                print(f"sending: {item}")
                client_socket.send(pack_msg(item))
        except BrokenPipeError:
            print("client disconnected")

def put_message(x: float, y: float, presence: int):
    # Convert presence to float for network transmission
    q.put_nowait((x, y, float(presence)))

def main():
    try:
        tcp_server_thread = Thread(target=start, args=(2025,))
        tcp_server_thread.start()
        
        # Example reading from neural network output
        # Replace this with actual neural network output reading
        while True:
            with open('your_output_file.txt', 'r') as f:
                line = f.readline().strip()
                if line:
                    # Parse "1 -1.0 3.0" format
                    presence, x, y = map(float, line.split())
                    put_message(x, y, presence)
            sleep(0.016)  # 60fps approximately
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()