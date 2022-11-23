import time
import serial

def arduino2():
    bag = []
    bag2 = []
    arduino = serial.Serial(port = "/dev/ttyACM0", baudrate = 115200)
    arduino_data = arduino.readline()
    arduino_data = arduino_data.decode() 
    arduino_data_list = arduino_data.split()
    if len(arduino_data_list)==2 :
        arduino_data_list= list(map(float,arduino_data_list))
        if arduino_data_list[0] <-1290:
            bag = ['shampoo','body wash', 'cleansing foam']
        elif arduino_data_list[0] >-200 and arduino_data_list[0] <-150:
            bag = ['body wash'] 
        elif arduino_data_list[0] >-130 and arduino_data_list[0] <-110:
            bag = ['cleansing foam']
        elif arduino_data_list[0] >-1020 and arduino_data_list[0] <-900:
            bag = ['shampoo'] 
        elif arduino_data_list[0] >-1150 and arduino_data_list[0] <-1100:
            bag = ['shampoo', 'cleansing foam']
        elif arduino_data_list[0] >-1200 and arduino_data_list[0] <-1170:
            bag = ['shampoo','body wash']
        elif arduino_data_list[0] >-320 and arduino_data_list[0] <-280:
            bag = ['body wash', 'cleansing foam']
        elif arduino_data_list[0] >-50:
            bag = 0
        if arduino_data_list[1] < -830:
            bag2 = ['coffee','tissue', 'rice']
        elif arduino_data_list[1] > -740 and arduino_data_list[1] < - 690:
            bag2 = ['coffee', 'rice']
        elif arduino_data_list[1] > -660 and arduino_data_list[1] < - 600:
            bag2 = ['coffee', 'tissue']
        elif arduino_data_list[1] > -520 and arduino_data_list[1] < -470:
            bag2 = ['coffee']
        elif arduino_data_list[1] > -380 and arduino_data_list[1] < - 330:
            bag2 = ['tissue', 'rice']
        elif arduino_data_list[1] > -260 and arduino_data_list[1] < - 200:
            bag2 = ['rice']
        elif arduino_data_list[1] > -160 and arduino_data_list[1] < - 100:
            bag2 = ['tissue']
        elif arduino_data_list[1] > -50:
            bag2 = 0
        return bag,bag2
    return 0,0