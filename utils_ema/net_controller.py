import socket
import logging
import subprocess


class NetController:

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # logging.basicConfig(level=logging.INFO)

    # @staticmethod
    # def check_ping(ip):
    #     response = ping(ip)
    #     if response is None:
    #         logging.error(f"{ip} is unreachable.")
    #         return None
    #     else:
    #         return f"{response:.2f}"

    @staticmethod
    def check_reachability(ip):
        try:
            output = subprocess.run(['ping', '-c', '4', ip], capture_output=True, text=True)
            if output.returncode == 0:
                return True
            else:
                logging.error(f"{ip} is not reachable.")
                return False
        except Exception as e:
            print(f"Failed to ping due to: {e}")

    # @staticmethod
    # def check_reachability(ip):
    #     res = NetController.check_ping(ip)
    #     if res is None: return False
    #     else: return True

    @classmethod
    def tcp_connect(cls, ip, port):
        try:
            cls.sock.connect((ip, port))
        except: pass

    @classmethod
    def send_tcp_message(cls, ip, port, message):

        # Create a TCP/IP socket
        cls.tcp_connect(ip, port)

        # try:

        # Send data
        cls.sock.sendall(message.encode('utf-8'))
        # logging.info("Message sent to {ip}:{port},"+ message)

        # Look for the response (optional)
        response = cls.sock.recv(1024)
        resp_string = response.decode('utf-8')
        # logging.info("Received: "+ resp_string)
        return resp_string

        # except:
        #     return None

        
    @classmethod
    def send_udp_message(cls, ip, port, message):

        # try:
        cls.tcp_connect(ip, port)

        # Convert message to bytes and send it to the specified IP and port
        cls.sock.sendto(message.encode(), (ip, port))
        logging.info("Message sent to {ip}:{port},"+ message)
        return True

        # except:
        #     return None

        
    @classmethod
    def close(cls):
        cls.sock.close()
