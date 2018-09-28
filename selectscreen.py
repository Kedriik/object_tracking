#thx Markacho

import win32api
import time
def select_screen():
    width = win32api.GetSystemMetrics(0)
    height = win32api.GetSystemMetrics(1)
    midWidth = int((width + 1) / 2)
    midHeight = int((height + 1) / 2)
    
    
    pos = []
    gen = 0
    state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
    while True:
        a = win32api.GetKeyState(0x01)
        if a != state_left:  # Button state changed
            state_left = a
            print(a)
            if a < 0:
                print('Corner 1 selected')
                x1,y1=win32api.GetCursorPos()
                pos.append(x1)
                pos.append(y1)
                print(pos)
                
            else:
                print('Corner 2 selected')
                x2,y2=win32api.GetCursorPos()
                #x2 = x2 - x1
                #y2 = y2 - y1
                pos.append(x2)
                pos.append(y2)
                print(pos)
                break
            time.sleep(0.001)
    return pos
        