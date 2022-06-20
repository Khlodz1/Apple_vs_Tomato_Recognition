import serial
gps = serial.Serial("COM5", baudrate=9600)
while(1):
    line = gps.readline()
    line = str(line)
    data = line.split(",")
    if("$GNRMC" in data[0]):
        data[3] = float(data[3])/100
        data[5] = float(data[5])/100
        print("lat: %f, long: %f"%(data[3],data[5]))
