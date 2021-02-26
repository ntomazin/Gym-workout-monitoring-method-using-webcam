# import everything from tkinter module
from tkinter import *
# create a tkinter window
root = Tk()

# Open window having dimension 100x100
root.geometry('640x480')

# Create a Button
btn = Button(root, text='Squat', bd='5',
             command=root.destroy, height=10,
             width=480)
btn2 = Button(root, text='Biceps', bd='5',
              command=root.destroy, height=10,
             width=480)
btn3 = Button(root, text='KettleBell', bd='5',
              command=root.destroy, height=10,
             width=480)

# Set the position of button on the top of window.
btn.pack(side='top')
btn2.pack(side='bottom')
btn3.pack(side='bottom')


root.mainloop()
