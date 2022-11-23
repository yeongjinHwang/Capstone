import time
import serial
arduino = serial.Serial(port = "/dev/ttyACM0", baudrate = 115200)
bag = []
bag2 = []
arduino = serial.Serial(port = "/dev/ttyACM0", baudrate = 115200)
while True :
    arduino_data = arduino.readline()
    arduino_data = arduino_data.decode()
    arduino_data_list = arduino_data.split()
    if len(arduino_data_list)==2 :
        arduino_data_list= list(map(float,arduino_data_list))
        if arduino_data_list[0] <-1290:
            bag = ['shampoo','body wash', 'cleansing foam']
        if arduino_data_list[0] >-200 and arduino_data_list[0] <-150:
            bag = ['body wash']
        if arduino_data_list[0] >-130 and arduino_data_list[0] <-110:
            bag = ['cleansing foam']
        if arduino_data_list[0] >-1020 and arduino_data_list[0] <-900:
            bag = ['shampoo']
        if arduino_data_list[0] >-1150 and arduino_data_list[0] <-1100:
            bag = ['shampoo', 'cleansing foam']
        if arduino_data_list[0] >-1200 and arduino_data_list[0] <-1170:
            bag = ['shampoo','body wash']
        if arduino_data_list[0] >-320 and arduino_data_list[0] <-280:
            bag = ['body wash', 'cleansing foam']
        if arduino_data_list[0] >-50:
            bag = 0
        bag2 = 0
        print(bag,bag2)
    else:
        print(0,0)