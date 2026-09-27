# % Wrapper class for GraphCut.dll to perform graph cut segmentation
# % ECE 4370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
#
from ctypes import*
import os
import numpy as np
# Instructions for recompiling GraphCut.dll
# From https://github.com/niXman/mingw-builds-binaries/releases
# Download and unzip x86_64-12.2.0-release-win32-sjlj-rt_v10-rev0.7z
# in the terminal cd to directory with graph cut cpp code
#cd C:\Users\noblejh\Box Sync\Code\GraphCut\GraphCut
# in the terminal do (replace path to the exe):
#C:\path\to\mingw64\bin\x86_64-w64-mingw32-g++.exe -o graphcut.dll -mdll -static-libstdc++ -static dllmain.cpp
#put the resulting dll file in the same directory as graphCut.py

# on Mac using clang++ or g++ from XCode
# g++ -dynamiclib -DNDEBUG -o graphcut.dylib dllmain.cpp
# A similar approach with different compile options should permit creating static .o library for linux using g++


class graphCut:
    def __init__(self, sigma=20, alpha=.975, lmbda=0.001):
        self.flow = 0
        self.sigma = sigma
        self.alpha = alpha
        self.lmbda = lmbda
        self.gptr = np.array(0)
        self.edgeweights = []
        self.cls = []
        self.mydll = cdll.LoadLibrary(os.getcwd()+"\\GraphCut.dll")
        self.mydll.Iter1.restype = c_void_p

    def updateSeeds(self,cls):
        if np.size(self.cls)==0:
            self.cls = np.zeros(np.flip(np.shape(cls)), dtype=np.uint8)
        self.cls[:,:,:] = np.swapaxes(cls,0,2)

    def updateTLinks(self,logpdf1,logpdf2):
        if np.size(self.edgeweights)==0:
            self.edgeweights = np.zeros(np.flip(np.insert(np.shape(logpdf1),0,5)),dtype=np.float32)

        tlogpdf1 = np.swapaxes(logpdf1,0,2)
        tlogpdf2 = np.swapaxes(logpdf2,0,2)
        self.edgeweights[:, :, :, 0] = -self.lmbda * tlogpdf2
        self.edgeweights[:, :, :, 1] = -self.lmbda * tlogpdf1

    def updateNLinks(self,img):
        dim = np.shape(img)
        if np.size(self.edgeweights)==0:
            self.edgeweights = np.zeros(np.flip(np.insert(dim,0,5)),dtype=np.float32)
        sigmasq = self.sigma*self.sigma
        wx = (np.exp(-(img[0:dim[0]-1,:,:]-img[1:dim[0],:,:]) / (2 * sigmasq) *
                                                      (img[0:dim[0]-1,:,:]-img[1:dim[0],:,:]) )
                                                * self.alpha + 1 - self.alpha)*(1-self.lmbda)

        self.edgeweights[:, :, 0:(dim[0] - 1), 2] = np.swapaxes(wx, 0, 2)
        wy = (np.exp(-(img[:,0:dim[1]-1,:]-img[:,1:dim[1],:]) / (2 * sigmasq) *
                                                      (img[:,0:dim[1]-1,:]-img[:,1:dim[1],:]) )
                                               * self.alpha + 1 - self.alpha)*(1 - self.lmbda)
        self.edgeweights[:, 0:(dim[1] - 1), :, 3] = np.swapaxes(wy, 0, 2)
        wz = (np.exp(-(img[:,:,0:dim[2]-1]-img[:,:,1:dim[2]]) / (2 * sigmasq) *
                                                      (img[:,:,0:dim[2]-1]-img[:,:,1:dim[2]]) )
                                                * self.alpha + 1 - self.alpha) * (1 - self.lmbda)
        self.edgeweights[0:(dim[2] - 1), :, :, 4] = np.swapaxes(wz, 0, 2)


    def run(self,changenodes=[]):
        v = np.flip(np.array(np.shape(self.edgeweights),np.int32))
        w = np.copy(v[1:])
        numchangenodes = np.array(np.size(changenodes), dtype=np.int32)
        if numchangenodes==0:
            if self.gptr != 0:
                self.mydll.Delete(self.gptr)
            self.gptr = c_void_p(self.mydll.Iter1(w.ctypes.data_as(POINTER(c_int32)),
                                         self.edgeweights.ctypes.data_as(POINTER(c_float)),
                                         self.cls.ctypes.data_as(POINTER(c_uint8))))
        else:
            self.mydll.Update(w.ctypes.data_as(POINTER(c_int32)),
                              self.edgeweights.ctypes.data_as(POINTER(c_float)),
                              self.cls.ctypes.data_as(POINTER(c_uint8)),
                              self.gptr,
                              changenodes.ctypes.data_as(POINTER(c_int32)),
                              c_int32(numchangenodes))
        return np.array(np.swapaxes(self.cls==1,0,2))

    def __del__(self):
        if self.gptr != 0:
            self.mydll.Delete(self.gptr)
            self.gptr = np.array(0)