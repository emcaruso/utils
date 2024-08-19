from utils_ema.net_controller import NetController
import time
import logging

class LightController():

    def __init__(self, ip_controller, protocol="tcp"):
        self.ip = ip_controller
        self.udp_port_in = 30312
        self.udp_port_out = 30313
        self.tcp_port_in = 30313
        self.tcp_port_out = 30312
        self.n_channels = 16
        self.protocol = protocol
        self.ampere_default = 0.01
        if not NetController.check_reachability(self.ip):
            raise ConnectionError("Light controller is not reachable!")
        else:
            logging.info("Light controller connection is working")

    def __send_message(self, message, log=False, protocol="tcp"):
        if self.protocol == "tcp":
            res = NetController.send_tcp_message(self.ip, self.tcp_port_in, message)
        elif self.protocol == "udp":
            res = NetController.send_udp_message(self.ip, self.tcp_port_in, message)
        else:
            raise ValueError(f"{self.protocol} is not a known protocol (tcp, udp)")

        if res is None:
            logging.error("Communication with light controller is not working!")
            return None

        if log:
            logging.info(res)
        return res

    # get status

    def log_channel_status(self, channel):
        res = self.__send_message("ST"+str(channel))
        logging.info("   channel: "+str(channel).zfill(2)+res.split("M")[1][:-3])

    def log_all_channel_status(self):
        for i in range(self.n_channels):
            self.log_channel_status(i)

    def log_trigger_status(self, log=True):
        res = self.__send_message("ST16")
        if log: logging.info("   "+res.split("ST16")[1][3:-3])
        return res

    def log_status(self):
        self.log_all_channel_status()
        self.log_trigger_status()

    # set triggers

    def set_default_trigger(self):
        self.__send_message("TT0")

    def set_trigger_groups(self, mode):
        self.__send_message("FP"+str(mode), protocol=self.protocol)

    def send_trigger_pulse(self, trigger_id):
        self.__send_message("TR"+str(trigger_id), protocol=self.protocol)

    # test trigger
    def set_trigger_test(self, milliseconds):
        self.__send_message("TT1"+","+str(milliseconds)+"MS", protocol=self.protocol)

    def test_lights_and_trigger(self, amp=0.0001, period_ms=40, pulse_width_ms=20, time_s=1):
        for i in range(self.n_channels):
            # self.set_led_pulse(channel=i, amp=amp, width_ms=pulse_width_ms)
            self.set_led_switch(channel=i, amp=amp)
        self.set_trigger_test(period_ms)
        time.sleep(time_s)
        self.clear_settings()


    # set leds

    def set_led_pulse(self, channel, amp=None, width_ms=200, trig_delay=0, retrig_delay=0):
        if amp is None: amp=self.ampere_default
        self.__send_message("RT"+str(channel)+","+str(width_ms*1000)+","+str(trig_delay)+","+str(amp)+","+str(retrig_delay), protocol=self.protocol)

    def set_led_continuous(self, channel, amp=None):
        if amp is None: amp=self.ampere_default
        self.__send_message("RS"+str(channel)+","+str(amp), protocol=self.protocol)

    def set_led_switch(self, channel, amp=None):
        if amp is None: amp=self.ampere_default
        self.__send_message("RW"+str(channel)+","+str(amp), protocol=self.protocol)

    def set_led_pulse_all(self, amp=None, width_ms=200, trig_delay=0, retrig_delay=0):
        if amp is None: amp=self.ampere_default
        for i in range(self.n_channels):
            self.set_led_pulse(i, amp=amp, width_ms=width_ms, trig_delay=trig_delay, retrig_delay=retrig_delay)

    def led_on(self, channel, amp=None, only=False):
        if amp is None: amp=self.ampere_default

        if only:
            for i in range(self.n_channels): self.led_off(i)

        self.set_led_continuous(channel, amp=amp)
 
    def led_off(self, channel):
        self.set_led_continuous(channel=channel, amp=0)
    # def led_pulse(self, channel, 


    # def led_pulse(self, channel, 


    def clear_settings(self):
        self.__send_message("CL", protocol=self.protocol)


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    # start_server('10.0.2.15', 30312)
    lc = LightController('192.168.1.100')

    # lc.set light(
    # lc.set_trigger_groups(0)
    # lc.set_trigger(0,200)
    # lc.send_trigger_pulse(0)

    lc.test_lights_and_trigger()
    # lc.clear_settings()                
    lc.log_status()
    # lc.test_lights_and_trigger()
    # time.sleep(10)
    # lc.clear_settings()


    # lc.send_trigger_pulse(0)
    # lc.clear_settings()



