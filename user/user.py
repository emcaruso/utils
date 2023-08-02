from pynput import keyboard


class user():
    # Mapping of keys to their corresponding names
    key_mapping = {
        keyboard.Key.up: "w",
        keyboard.Key.left: "a",
        keyboard.Key.down: "s",
        keyboard.Key.right: "d",
    }
    key = None

    @staticmethod
    def on_press(key):
        # key_name = user.key_mapping[key]
        user.key = key
        return False

    @staticmethod
    def detect_key():
        with keyboard.Listener(on_press=user.on_press) as listener:
            listener.join()
            return user.key


if __name__=="__main__":
    while True:
        user.detect_key()
        print(user.key)
