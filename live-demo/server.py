from time import sleep
from socket import socket
from threading import Thread
from typing import Optional
from queue import Queue
import struct

q = Queue()

def pack_msg(msg: tuple[int, int, int]) -> bytes:
    m = struct.pack("!iii", *msg)
    return bytes(m)

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
                print(f"got item in queue: {item}")

                client_socket.send(pack_msg(item))

        except BrokenPipeError:
            print("client disconnected")

def put_message(msg: tuple[int, int, int]):
    q.put_nowait(msg)

def main():
    try:
        tcp_server_thread = Thread(target=start, args=(2025,))

        tcp_server_thread.start()

        while True:
            put_message((-1, 3, 1))
            sleep(0.5)

    except:
        print("error")

if __name__ == "__main__":
    main()