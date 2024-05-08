import socket

def send_message(ip, port, message):
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the server's IP and port
    server_address = (ip, port)
    print(f"Connecting to {ip}:{port}")
    client_socket.connect(server_address)

    try:
        # Send data
        client_socket.sendall(message.encode('utf-8'))
        print("Message sent:", message)

        # Look for the response (optional)
        response = client_socket.recv(1024)
        print('Received:', response.decode('utf-8'))
    
    finally:
        # Clean up the connection
        client_socket.close()

def start_server(host='192.168.1.100', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        while True:
            client_socket, addr = server_socket.accept()
            with client_socket:
                print('Connected by', addr)
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    print("Received:", data.decode('utf-8'))
                print("Connection closed with", addr)

# Example usage
start_server()


if __name__=="__main__":
    start_server('10.0.2.15', 30312)
    send_message('10.0.2.15', 30313, 'Hello, Server!')



