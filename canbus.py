import can
def post(angle=90,speed=0,stop=1):
    print(angle,speed,stop)
    return
    bus = can.interface.Bus(channel='can0', interface='socketcan')

    msg = can.Message(arbitration_id=0x12, data=[angle, speed,stop], is_extended_id=False)

    try:
        bus.send(msg)
        print("Message sent")
    except can.CanError:
        print("Message failed")