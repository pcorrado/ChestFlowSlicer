import tkinter as tk
from PIL import Image, ImageTk

class SliceViewer(tk.Frame):
    def __init__(self,master,img,row,col,nx,ny,nSlice):
        self._defaultSize = 256
        self._scaleFactor = self._defaultSize/10.0
        self._planeInitialized = False

        tk.Frame.__init__(self,master, width=self._defaultSize, height=self._defaultSize)
        self.grid(row=row, column=col, sticky=tk.N+tk.S+tk.E+tk.W) #fill=tk.BOTH, expand=tk.YES, side=side)

        self.image = img
        self.nx = nx
        self.ny=ny
        self.nSlice = nSlice

        self.scale = tk.Scale(master=self, from_=0, to=(self.nSlice-1),width=12, length=self._defaultSize, bd=0)
        self.scale.pack(side=tk.RIGHT, fill=tk.Y, expand=tk.NO)

        self.update()
        self.canvas = tk.Canvas(master=self, width=self._defaultSize-self.scale.winfo_width(),height=self._defaultSize)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
        self.img_copy = self.image.copy().resize(size=(self._defaultSize-self.scale.winfo_width(),self._defaultSize), resample=Image.BILINEAR)
        self.background_image = ImageTk.PhotoImage(self.img_copy)
        self.imageObject = self.canvas.create_image(0,0, anchor="nw", image=self.background_image)
        self.xCrossingLine = self.canvas.create_line(0, int(self.canvas.winfo_width()/2), self.canvas.winfo_height(),int(self.canvas.winfo_width()/2), dash=(3,3),fill="yellow", width=1)
        self.yCrossingLine = self.canvas.create_line(int(self.canvas.winfo_height()/2), 0, int(self.canvas.winfo_height()/2), self.canvas.winfo_width(), dash=(3,3), fill="yellow",width=1)
        self.canvas.bind("<Button-1>", self._clickCallback)
        self.xCrossing = self.nSlice/2
        self.yCrossing = self.nSlice/2
        self.bind('<Configure>', self._resize_image)

        self.scale.set(round(nSlice/2))

        self.colors = ["red","yellow"]
        self.lineCoords = [0 for _ in range(len(self.colors))]
        self.arrowCoords = [0 for _ in range(len(self.colors))]
        self.line = [0 for _ in range(len(self.colors))]
        self.arrow = [0 for _ in range(len(self.colors))]

    def setSliderCallback(self,callback):
        self.sliderCallback = callback
        self.scale.config(command=self.sliderCallback)

    def setClickCallback(self,callback):
        self.clickCallback=callback

    def setImage(self,img):
        self.image = img
        self.updateUI()

    def setXCrossing(self,val):
        self.YCrossing = val
        y = int(self.canvas.winfo_width()*int(val)/self.nx)
        self.canvas.coords(self.xCrossingLine, (y,0,y,int(self.canvas.winfo_height())))

    def setYCrossing(self,val):
        self.xCrossing = val
        x = int(self.canvas.winfo_height()*int(val)/self.ny)
        self.canvas.coords(self.yCrossingLine, (0,x,int(self.canvas.winfo_width()),x))

    def _resize_image(self,event):
        new_width = event.width
        new_height = event.height
        self.updateUI(w=new_width, h=new_height)
        self._scaleFactor = min(new_width,new_height)/10.0

    def _clickCallback(self,event):
        x = int(event.x*self.nx/float(self.canvas.winfo_width()))
        y = int(event.y*self.ny/float(self.canvas.winfo_height()))
        self.clickCallback(x,y)

    def _lineMoveCallback(self,event):
        x = int(event.x*self.nx/float(self.canvas.winfo_width()))
        y = int(event.y*self.ny/float(self.canvas.winfo_height()))
        self.lineMoveCallback(x,y)
    def _arrowMoveCallback(self,event):
        x = int(event.x*self.nx/float(self.canvas.winfo_width()))
        y = int(event.y*self.ny/float(self.canvas.winfo_height()))
        self.arrowMoveCallback(x,y)


    def addPlane2(self,line,normal,lineCallback,arrowCallback,index=0):
        (x1,y1,x2,y2) = line
        x1*=self.canvas.winfo_width()/self.nx
        x2*=self.canvas.winfo_width()/self.nx
        y1*=self.canvas.winfo_height()/self.ny
        y2*=self.canvas.winfo_height()/self.ny
        self.lineCoords[index] = (x1,y1,x2,y2)
        midpointX = (x1+x2)/2
        midpointY = (y1+y2)/2
        (nx,ny) = normal
        self.arrowCoords[index] = (midpointX,midpointY,midpointX+nx*self._scaleFactor,midpointY+ny*self._scaleFactor)
        self.arrow[index] = [self.canvas.create_line(self.arrowCoords[index],fill=self.colors[index],arrow=tk.LAST, width=2, tags="arrow")]
        self.canvas.tag_bind(self.arrow[index], "<B1-Motion>", self._arrowMoveCallback)
        self.arrowMoveCallback = arrowCallback
        self.line[index] = [self.canvas.create_line(self.lineCoords[index],fill=self.colors[index], width=4, tags="line")]
        self.canvas.tag_bind(self.line[index], "<B1-Motion>", self._lineMoveCallback)
        self.lineMoveCallback = lineCallback
        self._planeInitialized = True

    def addPlane(self,line,normal,lineCallback,arrowCallback,index=0):
        (x1,y1,x2,y2) = line
        x1*=self.canvas.winfo_width()/self.nx
        x2*=self.canvas.winfo_width()/self.nx
        y1*=self.canvas.winfo_height()/self.ny
        y2*=self.canvas.winfo_height()/self.ny
        self.lineCoords = [(x1,y1,x2,y2)]
        midpointX = (x1+x2)/2
        midpointY = (y1+y2)/2
        (nx,ny) = normal
        self.arrowCoords = [(midpointX,midpointY,midpointX+nx*self._scaleFactor,midpointY+ny*self._scaleFactor)]
        self.arrow = [self.canvas.create_line(self.arrowCoords,fill=self.colors[index],arrow=tk.LAST, width=2, tags="arrow")]
        self.canvas.tag_bind(self.arrow, "<B1-Motion>", self._arrowMoveCallback)
        self.arrowMoveCallback = arrowCallback
        self.line = [self.canvas.create_line(self.lineCoords,fill=self.colors[index], width=4, tags="line")]
        self.canvas.tag_bind(self.line, "<B1-Motion>", self._lineMoveCallback)
        self.lineMoveCallback = lineCallback
        self._planeInitialized = True

    def movePlane(self,line,normal, index=0):
        (x1,y1,x2,y2) = line
        x1*=self.canvas.winfo_width()/self.nx
        x2*=self.canvas.winfo_width()/self.nx
        y1*=self.canvas.winfo_height()/self.ny
        y2*=self.canvas.winfo_height()/self.ny
        self.lineCoords[index] = (x1,y1,x2,y2)
        midpointX = (x1+x2)/2
        midpointY = (y1+y2)/2
        (nx,ny) = normal
        self.arrowCoords[index] = (midpointX,midpointY,midpointX+nx*self._scaleFactor,midpointY+ny*self._scaleFactor)
        self.canvas.coords(self.line[index], self.lineCoords[index])
        self.canvas.coords(self.arrow[index], self.arrowCoords[index])

    def removePlane(self):
        if self._planeInitialized:
            self.canvas.delete(self.line)
            self.canvas.delete(self.arrow)
            self._planeInitialized = False

    def updateUI(self,w=None,h=None):
        if w==None and h==None:
            w = self.canvas.winfo_width()+self.scale.winfo_width()
            h = self.canvas.winfo_height()
        self.update()
        self.img_copy = self.image.resize(size=(w-self.scale.winfo_width(), h), resample=Image.BILINEAR)
        self.background_image = ImageTk.PhotoImage(self.img_copy)
        self.canvas.itemconfig(self.imageObject,image=self.background_image)
