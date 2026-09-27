# % Class to perform interactive semi-automatic graph cut segmentation using graphCut.py and GraphCut.dll
# % ECE 4370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

from .volumeViewer import *
from .graphCut import *
import numpy as np
from matplotlib.widgets import Button
import matplotlib.pyplot as plt


class interactiveGraphcut:
    def __init__(self, img, voxsz,sigma = 500,lmbda = 0.05,alpha = 0.9,
                 rad = 10,contrast=1,level=0,seed_fore = [],seed_back = [], autocontrast=True):
        self.img = img
        self.voxsz = voxsz
        self.sigma = sigma
        self.lmbda = lmbda
        self.alpha = alpha
        self.rad = rad
        self.seed_fore = np.copy(seed_fore)
        self.seed_back = np.copy(seed_back)
        self.seed_fore_prev = []
        self.seed_back_prev = []
        self.pdffore = []
        self.pdfback = []
        self.d = volumeViewer()
        self.d.setImage(self.img, self.voxsz, contrast=contrast, level=level, autocontrast=autocontrast)
        self.d.update()
        if np.size(self.seed_fore)==0:
            self.seed_fore = np.zeros(np.shape(img), dtype=np.uint8)
        else:
            self.d.addMask(self.seed_fore, color=[0,1,0], opacity=0.5, label='foreground')
        if np.size(self.seed_back)==0:
            self.seed_back = np.zeros(np.shape(img), dtype=np.uint8)
        else:
            self.d.addMask(self.seed_back, color=[1,0,0], opacity=0.5, label='background')
        self.seg = np.zeros(np.shape(img))

        dx = np.ceil(rad/voxsz[0]).astype(np.int32)
        dy = np.ceil(rad/voxsz[1]).astype(np.int32)
        dz = np.ceil(rad/voxsz[2]).astype(np.int32)
        X, Y, Z = np.meshgrid(np.linspace(-dx, dx, 2*dx+1),
                              np.linspace(-dy, dy, 2*dy+1),
                              np.linspace(-dz, dz, 2*dz+1))
        self.se = voxsz[0]*voxsz[0]*X*X +voxsz[1]*voxsz[1]*Y*Y +voxsz[2]*voxsz[2]*Z*Z < rad*rad

        self.gc = graphCut(sigma,alpha,lmbda)

        mn = np.amin(img)
        mx = np.amax(img)
        self.bins=np.linspace(mn,mx,65)
        self.imgbin = np.floor(63.999*(img - mn)/(mx-mn)).astype(np.int32)
        self.hist_updated=0
        self.mode=0

        self.gc.updateNLinks(img)

        ax1 = plt.axes([0.55, 0.4, 0.4, 0.075])
        self.b1 = Button(ax1, 'Update Foreground Seeds')
        self.b1.on_clicked(self.updateForegroundSeeds)

        ax2 = plt.axes([0.55, 0.3, 0.4, 0.075])
        self.b2 = Button(ax2, 'Update Background Seeds')
        self.b2.on_clicked(self.updateBackgroundSeeds)

        ax3 = plt.axes([0.55, 0.2, 0.4, 0.075])
        self.b3 = Button(ax3, 'Update Histograms')
        self.b3.on_clicked(self.updateHistograms)

        ax4 = plt.axes([0.55, 0.1, 0.4, 0.075])
        self.b4 = Button(ax4, '(Re)compute Graph Cut')
        self.b4.on_clicked(self.runGraphCut)

        ax5 = plt.axes([0.55, 0.025, 0.4, 0.05])
        self.b5 = Button(ax5, 'Exit')
        self.b5.on_clicked(self.exit)

        axtxt = plt.axes([0.5, 0.03, 0.05, 0.45])
        plt.axis('off')
        self.ht = plt.text(0.1,.1,'*')
        self.ht.set(visible=False)

        plt.connect('button_press_event',self.onMouseClick)
        self.graphcut_output = []

        self.fig, self.ax = plt.subplots()
        plt.title('Fore and background histograms')

        self.d.display()




    def updateForegroundSeeds(self,event):
        self.mode = 1
        self.ht.set_y(0.9)
        self.ht.set(visible=True)

    def updateBackgroundSeeds(self,event):
        self.mode = 2
        self.ht.set_y(0.7)
        self.ht.set(visible=True)

    def updateHistograms(self,event):
        self.mode=3
        self.hist_updated=1
        self.ht.set_y(0.45)
        self.ht.set(visible=True)
        histfore=[]
        histback=[]
        for i in range(len(self.d.objs)):
            if self.d.objs[i].type == 1:
                if self.d.objs[i].data.label == 'foreground':
                    histfore, x = np.histogram(self.img[self.d.objs[i].data.data>0].flatten(),bins=self.bins)
                if self.d.objs[i].data.label == 'background':
                    histback, x = np.histogram(self.img[self.d.objs[i].data.data>0].flatten(),bins=self.bins)
        binc = (self.bins[0:np.size(self.bins)-1] + self.bins[1:])/2
        plt.figure(self.fig)
        plt.axes(self.ax)
        plt.cla()
        if np.size(histfore)>0:
            histfore = histfore / np.sum(histfore)
            if np.size(histback)>0:
                histback = histback / np .sum(histback)
                tot = histfore + histback
                tot[tot==0]=1
                self.pdffore = histfore/tot
                self.pdfback = histback/tot
                self.pdffore[self.pdffore<0.001]=0.001
                self.pdfback[self.pdfback<0.001]=0.001

                plt.plot(binc, self.pdffore, 'r', label='foreground cond. probability')
                plt.plot(binc, self.pdfback, 'b', label='background cond. probability')
                self.gc.updateTLinks(np.log(self.pdffore[self.imgbin]),np.log(self.pdfback[self.imgbin]))
            else:
                plt.plot(binc,histfore,'r',label='foreground pdf')
        elif np.size(histback)>0:
            histback = histback / np.sum(histback)
            plt.plot(binc, histback, 'b', label='background pdf')
        plt.yscale('log')
        plt.legend()
        plt.axis([self.bins[0], self.bins[-1],.001,1])
        plt.show()
        plt.figure(self.d.fig)
    def runGraphCut(self,event):
        if self.hist_updated<1:
            return
        self.mode = 4
        self.ht.set_y(0.2)
        self.ht.set(visible=True)

        cls = np.copy(self.seed_fore)
        cls[cls==0] = 2*self.seed_back[cls==0]
        self.gc.updateSeeds(cls)

        if self.hist_updated==2 and np.size(self.seed_fore_prev)>0:
            forechange = np.flatnonzero(np.swapaxes(np.not_equal(self.seed_fore_prev, self.seed_fore), 0, 2))
            backchange = np.flatnonzero(np.swapaxes(np.not_equal(self.seed_back_prev, self.seed_back), 0, 2))

        if self.hist_updated==1:
            self.gc.run()
        elif self.hist_updated==2:
            self.gc.run(np.concatenate((forechange,backchange)).astype(np.int32))
        self.hist_updated=2
        self.seg = np.array(np.swapaxes(self.gc.cls==1,0,2))
        fnd = 0
        for i in range(len(self.d.objs)):
            if self.d.objs[i].type == 1:
                if self.d.objs[i].data.label == 'segmentation':
                    plt.axes(self.d.ax[0, 0])
                    self.d.objs[i].data.data = self.seg
                    self.d.objs[i].data.updateContours(self.d)
                    self.d.update()
                    fnd = 1
                    break
        if fnd == 0:
            plt.axes(self.d.ax[0, 0])
            self.d.addMask(self.seg, color=[0,1,1], opacity=0.5, label='segmentation')
            self.d.update()
        self.d.update(3)
        self.seed_fore_prev = np.copy(self.seed_fore)
        self.seed_back_prev = np.copy(self.seed_back)

    def exit(self,event):
        plt.close(self.fig)
        self.d.quit=True
        self.graphcut_output=self.gc

    def onMouseClick(self,event):
        if event.dblclick or (self.mode!=1 and self.mode !=2):
            return
        win = -1
        for i in range(0,3):
            if event.inaxes == self.d.ax[i//2,i%2]:
                win=i
                break
        if win == 2:
            pnt = [event.xdata, event.ydata, self.d.slc[2]]
        elif win == 1:
            pnt = [event.xdata, self.d.slc[1], event.ydata]
        elif win == 0:
            pnt = [self.d.slc[0], event.xdata, event.ydata]
        else:
            return

        if self.mode==1:
            msk = self.seed_fore
            label='foreground'
            color = [0,1,0]
        else:
            msk = self.seed_back
            label='background'
            color = [1,0,0]
        dm = np.floor(np.array(np.shape(self.se))/2.0)
        mnsei = -(np.round(pnt)-dm)
        mnse = np.zeros(3,dtype=np.int32)
        mnse[mnsei>0] = mnsei[mnsei>0]
        szimg = np.array(np.shape(self.img),dtype=np.int32)
        mxsei = np.round(pnt)+dm-(szimg-1)
        mxse = np.array(np.shape(self.se),dtype=np.int32)
        mxse[mxsei>0] = mxse[mxsei>0] - mxsei[mxsei>0]
        mn = np.array(np.round(pnt)-dm,dtype=np.int32)
        mn[mn<0]=0
        mx = np.array(np.round(pnt)+dm+1,dtype=np.int32)
        mx[mx>szimg] = szimg[mx>szimg]
        if event.button == MouseButton.LEFT:
            msk[mn[0]:mx[0],mn[1]:mx[1],mn[2]:mx[2]] |= self.se[mnse[0]:mxse[0], mnse[1]:mxse[1], mnse[2]:mxse[2]]
        elif event.button == MouseButton.RIGHT:
            msk[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]] &= np.invert(self.se[mnse[0]:mxse[0], mnse[1]:mxse[1], mnse[2]:mxse[2]])
        fnd=0
        for i in range(len(self.d.objs)):
            if self.d.objs[i].data.label == label:
                plt.axes(self.d.ax[0,0])
                self.d.objs[i].data.updateContours(self.d)

                self.d.update()
                fnd=1
                break
        if fnd==0:
            plt.axes(event.inaxes)
            self.d.addMask(msk, color = color, opacity = 0.5, label=label)
            self.d.update()