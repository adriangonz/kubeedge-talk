import paho.mqtt.client as mqtt

# WIP


class LedMapper:
    def __init__(self, client_id="eventbus"):
        self._mqtt = mqtt.Client(client_id=client_id)
        self._mqtt.tls_insecure_set(True)

        self._mqtt.connect()


def main():
    pass


if __name__ == "__main__":
    main()
