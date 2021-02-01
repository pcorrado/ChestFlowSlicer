import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class CutPlaneViewer(tk.Frame):
    def __init__(self,master,img,row,col,plusCallback=None,minusCallback=None):
        self._defaultSize=256
        self._defaultHeight = 1
        tk.Frame.__init__(self,master)
        self.grid(row=row, column=col, sticky=tk.N+tk.S+tk.E+tk.W) #fill=tk.BOTH, expand=tk.YES, side=side)

        self.image = img

        # tk.Grid.rowconfigure(self, 0, weight=0)
        # tk.Grid.rowconfigure(self, 1, weight=1)
        # tk.Grid.columnconfigure(self, 0, weight=1)
        # tk.Grid.columnconfigure(self, 1, weight=1)

        self.buttonFrame = tk.Frame(master=self)
        self.buttonFrame.pack(side=tk.TOP,fill=tk.BOTH, expand=tk.NO)

        self.plusButton = tk.Button(master=self.buttonFrame, text="+", height=self._defaultHeight, command=plusCallback)
        #self.plusButton.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.plusButton.pack(side=tk.RIGHT,fill=tk.X,expand=tk.YES)
        self.minusButton = tk.Button(master=self.buttonFrame, text="-", height=self._defaultHeight, command=minusCallback)
        self.minusButton.pack(side=tk.RIGHT,fill=tk.X,expand=tk.YES)
        #self.minusButton.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.update()
        self.canvas = tk.Canvas(master=self,width=self._defaultSize, height=(self._defaultHeight-self.buttonFrame.winfo_height()))
        #self.canvas.grid(row=1,column=0,columnspan=2)
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)
        self.img_copy = self.image.copy()
        self.background_image = ImageTk.PhotoImage(self.img_copy)
        self.imageObject = self.canvas.create_image(0,0, anchor="nw", image=self.background_image)
        self.bind('<Configure>', self._resizeImage)

    def _resizeImage(self,event):
        self.updateUI()

    def setImage(self,img=None,xDir=None,yDir=None,scrollCallback=None):
        self.xDir=xDir
        self.yDir=yDir
        self.canvas.unbind('<Button-1>')
        self.scrollCallback = scrollCallback
        if not scrollCallback is None:
            self.canvas.bind('<Button-1>',self.buttonPressCallback)
            self.canvas.bind('<ButtonRelease-1>',self.buttonReleaseCallback)
        if img is None:
            self.image = Image.fromarray(np.zeros((50,50)))
        else:
            self.image = img
        self.updateUI()

    def buttonPressCallback(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.bind('<B1-Motion>',self.scroll)

    def scroll(self, event):
        (x1,x2,x3) = self.xDir
        (y1,y2,y3) = self.yDir
        dx = -(event.x-self.x)*x1 - (event.y-self.y)*y1
        dy = -(event.x-self.x)*x2 - (event.y-self.y)*y2
        dz = -(event.x-self.x)*x3 - (event.y-self.y)*y3
        self.scrollCallback(dx/50.0,dy/50.0,dz/50.0)
        self.x = event.x
        self.y = event.y

    def buttonReleaseCallback(self, event):
        self.canvas.unbind('<B1-Motion>')

    def updateUI(self):
        self.update()
        self.img_copy = self.image.resize(size=(self.canvas.winfo_width(),self.canvas.winfo_height()), resample=Image.BILINEAR)
        self.background_image = ImageTk.PhotoImage(self.img_copy)
        self.canvas.itemconfig(self.imageObject,image=self.background_image)
