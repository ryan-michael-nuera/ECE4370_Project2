# % Class to create interactive 3D VTK render window
# % ECE 5370: Engineering for Surgery
# % Fall 2025
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu
#

# % Example usage shown in the following demo functions below:
#  demoPointsAndLines()
#  demoSurfaceAppearance()
#  demoSurfaceEdgesAndColors()
#  demoDepthOfField()
#  brainPointPick()
#  bouncingBallsAnimation()
#  brainAnimation()


import vtk
import numpy as np

class vtkObject:
    def __init__(self, pnts=None, poly=None, actor=None):
        self.pnts = pnts
        self.poly = poly
        self.actor = actor

    def updateActor(self, verts):
        for j,p in enumerate(verts):
            self.pnts.InsertPoint(j,p)
        self.poly.Modified()


def ActorDecorator(func):
    def inner(verts,faces=None,color=[1,0,0],opacity=1.0, colortable=None, coloridx=None):
        pnts = vtk.vtkPoints()
        for j,p in enumerate(verts):
            pnts.InsertPoint(j,p)

        poly = func(pnts,faces)

        #important for smooth rendering
        norm = vtk.vtkPolyDataNormals()
        norm.SetInputData(poly)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(norm.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if coloridx is None:
            actor.GetProperty().SetColor(color[0],color[1],color[2])
        else:
            scalars = vtk.vtkDoubleArray()
            for j in range(len(verts)):
                scalars.InsertNextValue(coloridx[j] / (len(colortable)-1))

            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(len(colortable))
            for j in range(len(colortable)):
                lut.SetTableValue(j,colortable[j,0],colortable[j,1], colortable[j,2])

            lut.Build()

            poly.GetPointData().SetScalars(scalars)
            norm.SetInputData(poly)
            mapper.SetInputConnection(norm.GetOutputPort())
            prop = actor.GetProperty()
            # prop.SetColor(0,0,0)
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange([0.0, 1.0])

        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetPointSize(4)
        obj = vtkObject(pnts, poly, actor)
        return obj

    return inner

@ActorDecorator
def pointActor(pnts, faces=None):
    cells = vtk.vtkCellArray()
    for j in range(pnts.GetNumberOfPoints()):
        vil = vtk.vtkIdList()
        vil.InsertNextId(j)
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetVerts(cells)

    return poly

@ActorDecorator
def linesActor(pnts,lines):
    cells = vtk.vtkCellArray()
    for j, f in enumerate(lines):
        vil = vtk.vtkIdList()
        vil.InsertNextId(lines[j,0])
        vil.InsertNextId(lines[j,1])
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetLines(cells)

    return poly

@ActorDecorator
def surfActor(pnts,faces):
    cells = vtk.vtkCellArray()
    for j, f in enumerate(faces):
        vil = vtk.vtkIdList()
        vil.InsertNextId(faces[j,0])
        vil.InsertNextId(faces[j,1])
        vil.InsertNextId(faces[j,2])
        cells.InsertNextCell(vil)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pnts)
    poly.SetPolys(cells)

    poly.BuildCells()
    poly.BuildLinks()

    return poly



class vtkWin(vtk.vtkRenderer):
    def __init__(self, sizex=512, sizey=512, title="3D Viewer (press q to quit)"):
        super().__init__()
        self.renwin = vtk.vtkRenderWindow() #creates a new window
        self.renwin.SetWindowName(title)
        self.renwin.AddRenderer(self)
        self.renwin.SetSize(sizex, sizey)
        self.inter = vtk.vtkRenderWindowInteractor() #makes the renderer interactive
        self.inter.AddObserver('KeyPressEvent',self.keypress_callback,1.0)
        self.lastpickpos = np.zeros(3)
        self.lastpickcell = -1
        self.inter.SetRenderWindow(self.renwin)
        self.inter.Initialize()
        self.inter.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.objlist = []

        self.renwin.Render() # paints the window on the screen once

    def __del__(self):
        del self.renwin, self.inter


    def addPoints(self, verts, color=[1.,0.,0.], opacity=1.):
        obj = pointActor(np.asarray(verts), color=color, opacity=opacity)
        self.objlist.append(obj)
        self.AddActor(obj.actor)

    def addLines(self, verts, lns, color=[1.,0.,0.], opacity=1.):
        obj = linesActor(np.asarray(verts), np.asarray(lns), color=color, opacity=opacity)
        self.objlist.append(obj)
        self.AddActor(obj.actor)

    def addSurf(self, verts, faces, color=[1.,0.,0.], opacity=1.,
                specular=0.9, specularPower=25.0, diffuse=0.6, ambient=0, edgeColor=None,
                colortable=None, coloridx=None):
        obj = surfActor(np.asarray(verts), np.asarray(faces), color=color, opacity=opacity, colortable=colortable, coloridx=coloridx)
        self.objlist.append(obj)
        actor = obj.actor
        if edgeColor is not None:
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(edgeColor[0], edgeColor[1], edgeColor[2])
        actor.GetProperty().SetAmbientColor(color[0], color[1], color[2])
        actor.GetProperty().SetDiffuseColor(color[0], color[1], color[2])
        actor.GetProperty().SetSpecularColor(1.0,1.0,1.0)
        actor.GetProperty().SetSpecular(specular)
        actor.GetProperty().SetDiffuse(diffuse)
        actor.GetProperty().SetAmbient(ambient)
        actor.GetProperty().SetSpecularPower(specularPower)

        self.AddActor(actor)
        if len(self.objlist)==1:
            mn = actor.GetCenter()
            self.GetActiveCamera().SetFocalPoint(mn[0],mn[1],mn[2])

    def keypress_callback(self,obj,ev):
        key = obj.GetKeySym()
        if (key == 'u' or key == 'U'):
            pos = obj.GetEventPosition()

            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.0005)

            picker.Pick(pos[0],pos[1],0,self)

            self.lastpickpos = picker.GetPickPosition()
            self.lastpickcell = picker.GetCellId()
        return key

    def updateActor(self, id, verts):
        self.objlist[id].updateActor(np.asarray(verts))

    def cameraPosition(self, position=None, viewup=None, fp=None , focaldisk=None):
        cam = self.GetActiveCamera()
        if position is not None:
            cam.SetPosition(position[0], position[1], position[2])
        if viewup is not None:
            cam.SetViewUp(viewup[0], viewup[1], viewup[2])
        if fp is not None:
            cam.SetFocalPoint(fp[0], fp[1], fp[2])
        if focaldisk is not None:
            dist  = np.sqrt(np.sum((np.array(cam.GetFocalPoint()) - np.array(cam.GetPosition()))**2))
            cam.SetFocalDisk(focaldisk*dist)

    def render(self):
        self.ResetCameraClippingRange()
        self.renwin.Render()
        self.inter.ProcessEvents()

    def start(self):
        self.inter.Start()

# function to build cylindrical triangular surface mesh using two endpoints
def cylinder(vert1, vert2, rad=1.0, numcirc=16):
    verts = np.zeros((numcirc*2, 3))
    v = vert2 - vert1
    vec = np.array([1.0,0.,0.])
    if np.abs(np.sum(v*vec)/np.linalg.norm(v))>0.95:
        vec = np.array([0, 1.0,0.])

    v1 = np.cross(v, vec)[np.newaxis,:]
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(v, v1)[np.newaxis,:]
    v2 /= np.linalg.norm(v2)
    theta = np.linspace(0, 2*np.pi, numcirc)[:,np.newaxis]
    verts[0:numcirc,:] = vert1[np.newaxis,:] + rad*(np.cos(theta)*v1 + np.sin(theta)*v2)
    verts[numcirc::,:] = vert2[np.newaxis,:] + rad * (np.cos(theta) * v1 + np.sin(theta) * v2)

    faces = np.zeros((numcirc*2 + 2*(numcirc-2), 3), dtype=int)
    for i in range(numcirc-2):
        faces[i,:] = np.array([0, i+1, i+2])
    for i in range(numcirc-2):
        faces[i+numcirc-2,:] = np.array([0, i+1, i+2]) + numcirc
    for i in range(numcirc):
        faces[i+2*(numcirc-2),:] = np.array([i, (i+1)%numcirc, i+numcirc])
    for i in range(numcirc):
        faces[i+numcirc+2*(numcirc-2),:] = np.array([(i+1)%numcirc, (i+1)%numcirc+numcirc, i+numcirc, ])

    return verts, faces

# Basic point and line display
def demoPointsAndLines():
    verts = np.array([[0.,0.,0],[1.,1.,1.]])
    win = vtkWin(title="Two points and Three lines")
    win.addPoints(verts)
    win.cameraPosition(position=[0.,0.,5.],viewup=[0,1,0],fp=[0.5,.5,.5])

    #show three lines
    verts = np.array([[0.,0.,0],[1.,1.,1.],[1.,0.,0.]])
    lns = np.array([[0,1],[1,2],[2,0]])

    win.addLines(verts,lns,color=[0,0,1.])
    win.cameraPosition([0.,0.,5.],[0,1,0],[0.5,.5,.5])
    win.start()

# Different types of surface rendering
def demoSurfaceAppearance():
    verts = np.array([[0.,0.,0],[1.,1.,1.],[1.,0.,0.]])
    win = vtkWin(title='Ambient, diffuse, and specular rendering')

    # display surface
    sverts,sfaces = cylinder(verts[0,:],verts[1,:],rad=0.1,numcirc=16)
    win.addSurf(sverts,sfaces,color=[.5,.5,.5],opacity=1,specular=.1)

    sverts,sfaces = cylinder(verts[1,:],verts[2,:],rad=0.1,numcirc=32)
    win.addSurf(sverts,sfaces,color=[.5,.5,.5],opacity=1,specular=0,diffuse=0,ambient=1)

    sverts,sfaces = cylinder(verts[2,:],verts[0,:],rad=0.1,numcirc=32)
    win.addSurf(sverts,sfaces,color=[.5,.5,.5],opacity=1,specular=.9)

    win.cameraPosition([0.,0.,5.],[0,1,0],[0.5,.5,.5])
    win.start()


# Triangle edges can be made visible for wire display
def demoSurfaceEdgesAndColors():
    verts = np.array([[0.,0.,0],[0.,0.,1.]])
    win = vtkWin(title='Edge visibility/Colormapping')

    # display surface
    sverts,sfaces = cylinder(verts[0,:],verts[1,:],rad=0.1,numcirc=16)

    colortable = np.concatenate((
        np.concatenate((np.zeros(32),np.linspace(0.0,1.0,32)))[:,np.newaxis],  # red
        np.concatenate((np.linspace(0.0,1.0,32),np.linspace(1.0,0.0,32)))[:,np.newaxis],  #green
        np.concatenate((np.linspace(1.0,0.0,33)[1::],np.zeros(32)))[:,np.newaxis]),axis=1)
    mn = np.min(sverts[:,0])
    mx = np.max(sverts[:,0])
    coloridx = np.floor((sverts[:,0] - mn) / (mx - mn) * 63.999).astype(int)

    win.addSurf(sverts,sfaces,ambient=0.9, opacity=1, edgeColor=[0.,0.,0.],colortable=colortable,coloridx=coloridx)

    win.cameraPosition([5.,0.,.5],[0,0,1],[0,0,.5])
    win.start()

# Can simulate realistic camera optic effects using depth-of-field
def demoDepthOfField():
    verts = np.array([[0.,0.,0],[1.,1.,1.],[1.,0.,0.]])
    win = vtkWin(title='Simulating real lens depth-of-field')

    # display surface
    sverts,sfaces = cylinder(verts[0,:],verts[1,:],rad=0.1,numcirc=16)
    win.addSurf(sverts,sfaces,color=[.5,.5,.5],opacity=1,specular=.1)

    sverts,sfaces = cylinder(verts[1,:],verts[2,:],rad=0.1,numcirc=32)
    win.addSurf(sverts,sfaces,color=[.5,.5,.5],opacity=1,specular=0,diffuse=0,ambient=1)

    sverts,sfaces = cylinder(verts[2,:],verts[0,:],rad=0.1,numcirc=32)
    win.addSurf(sverts,sfaces,color=[.5,.5,.5],opacity=1,specular=.9)

    basicPasses = vtk.vtkRenderStepsPass()
    dofp = vtk.vtkDepthOfFieldPass()
    dofp.SetDelegatePass(basicPasses)
    dofp.AutomaticFocalDistanceOff()
    win.SetPass(dofp)

    # small focal disk -> longer depth of field
    win.cameraPosition(fp=[-1,-1,-1],focaldisk=.02, position=[-4, -2.5, -4], viewup=[0.25, 0.76, -0.6])
    win.start()


# Custom Point/Cell picking implemented with 'u' key
def brainPointPick():
    import json
    f = open('Demo/brain.json','rt')
    dct = json.load(f)
    f.close()
    verts = np.array(dct['verts'])
    faces = np.array(dct['faces'])

    class printPickWin(vtkWin):
        def keypress_callback(self,obj,ev):
            super().keypress_callback(obj,ev)
            worldPosition = self.lastpickpos
            cell = self.lastpickcell
            print(f'Picked point coordinate: {worldPosition[0]:.2f} {worldPosition[1]:.2f} {worldPosition[2]:.2f}')
            print(f'Cell Id: {cell:d}')
            cam = self.GetActiveCamera()
            campos = cam.GetPosition()
            camfp = cam.GetFocalPoint()
            camvu = cam.GetViewUp()
            print(f'Camera Position: {campos[0]:.2f} {campos[1]:.2f} {campos[2]:.2f}')
            print(f'Camera Focal Point: {camfp[0]:.2f} {camfp[1]:.2f} {camfp[2]:.2f}')
            print(f'Camera View Up: {camvu[0]:.2f} {camvu[1]:.2f} {camvu[2]:.2f}')

    win = printPickWin(1024,512, title='Point pick using ''u'' key')
    win.addSurf(verts,faces,color=[1.,.8,.8])
    vu = np.array([-.43,-.9,-.12])
    vu = vu / np.linalg.norm(vu)
    fp = np.mean(verts,axis=0)
    win.cameraPosition(position=[500,-40,15],viewup=vu,fp=fp)

    # try point picking with 'u'
    win.start()

# create screenshot test.png and video file test.avi with spinning brain using ffmpeg
# shows how to (1) move camera, (2) create screenshot, (3) create videos
def brainAnimation():
    import json
    import vtkmodules.vtkRenderingCore
    from subprocess import Popen,PIPE
    from vtk.util.numpy_support import vtk_to_numpy

    f = open('Demo/brain.json','rt')
    dct = json.load(f)
    f.close()
    verts = np.array(dct['verts'])
    faces = np.array(dct['faces'])

    win = vtkWin(1024,512,title='Screenshot and Video using ffmpeg')
    win.addSurf(verts,faces,color=[1.,.8,.8])
    vu = np.array([-.43,-.9,-.12])
    vu = vu / np.linalg.norm(vu)
    fp = np.mean(verts,axis=0)
    win.cameraPosition(position=[500,-40,15],viewup=vu,fp=fp)
    win.render()

    windowToImageFilter = vtkmodules.vtkRenderingCore.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(win.renwin)
    windowToImageFilter.SetInputBufferTypeToRGBA()
    windowToImageFilter.ReadFrontBufferOn()
    windowToImageFilter.Update()
    out = windowToImageFilter.GetOutput()

    png = vtk.vtkPNGWriter()
    png.SetInputData(out)
    png.SetFileName("test.png")
    png.Write()


    fps = 15
    N = 100
    cam = win.GetActiveCamera()
    command = ["C:\\Users\\noblejh\\Downloads\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg",
               '-loglevel','error',
               '-y',
               # Input
               '-f','rawvideo',
               '-vcodec','rawvideo',
               '-pix_fmt','bgr24',
               '-s',str(1024) + 'x' + str(512),
               '-r',str(fps),
               # Output
               '-i','-',
               '-an',
               '-vcodec','mpeg4',  #'h264',
               '-r',str(fps),
               '-pix_fmt','bgr24',
               "test.avi"
               ]
    p = Popen(command,stdin=PIPE)
    #timing looks rough in real time rendering but is fine in the final avi file
    for i in range(N):
        cam.Azimuth(360.0 / N)  # degrees
        win.render()
        windowToImageFilter = vtkmodules.vtkRenderingCore.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(win.renwin)
        windowToImageFilter.SetInputBufferTypeToRGBA()
        windowToImageFilter.ReadFrontBufferOff()

        windowToImageFilter.Update()
        out = windowToImageFilter.GetOutput()
        sc = out.GetPointData().GetScalars()
        r = vtk_to_numpy(sc)
        r2 = np.flip(np.flip(r.reshape(512,1024,4)[:,:,0:3],axis=2),axis=0)
        r2o = r2.tobytes()
        p.stdin.write(r2o)

    p.stdin.close()
    p.wait()

    win.start()


# shows how to (1) create surface using marching cubes,
# (2) manipulate surfaces for animations, (3) create custom lighting/shadows
def bouncingBallsAnimation():
    import skimage.measure
    import vtkmodules.vtkRenderingCore
    N = 1000
    rad1 = 1
    rad2 = .5

    # sphere equation on grid
    X,Y,Z = np.meshgrid(np.arange(-25,26), np.arange(-25,26), np.arange(-25,26), indexing='ij')
    sph = 400 - (X*X +Y*Y + Z*Z)

    # sphere centered at [25,25,25] with radius=20 voxels
    verts, faces, _, _ = skimage.measure.marching_cubes(sph, 0)

    # zero center and normalize radius to 1
    verts = (verts - 25)/ 20

    #create 2 side-by-side spheres
    sph1 = verts*rad1
    sph2 = verts*rad2 + np.array([[2.,0.,0.]])

    # create 'floor' to bounce the spheres on
    vertsfloor = np.array([[-2,-5,0],[6,-5,0],[-2,5,0],[6,5,0]])
    trisfloor = np.array([[0,1,2],[2,1,3]],dtype=int)

    win = vtkWin(512,512,title='bouncing balls')

    shadows = vtk.vtkShadowMapPass()
    seq = vtk.vtkSequencePass()

    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(shadows.GetShadowMapBakerPass())
    passes.AddItem(shadows)
    seq.SetPasses(passes)

    cameraP = vtk.vtkCameraPass()
    cameraP.SetDelegatePass(seq)

    # Tell the renderer to use our render pass pipeline
    win.SetPass(cameraP)

    win.addSurf(sph1, faces, color=[1,0,0], specular=0.9)
    win.addSurf(sph2, faces, color=[0,1,0], specular=0.9)
    win.addSurf(vertsfloor,trisfloor,color=[1,1,1],ambient=0.2)
    win.cameraPosition(position=[1.5,-15,4],viewup=[0,0,1],fp=[1.5,0,1])

# create static light
    light = vtk.vtkLight()
    light.SetFocalPoint(2.5,0,0)
    light.SetPosition(-15,0,20)
    win.AddLight(light)
    cam = win.GetActiveCamera()

    theta = np.linspace(0,np.pi,50)
    for i in range(N):
        sph1[:,2] = verts[:,2]*rad1 + rad1 + np.sin(theta[i % 50])
        sph2[:,2] = verts[:,2]*rad2 + rad2 + np.sin(theta[(i+25) % 50])
        win.updateActor(0, sph1)
        win.updateActor(1, sph2)
        cam.Azimuth(360.0 / N)
        win.render()


    win.start()


# Surface class can contain the verts/faces and we can build member functions for surface analysis
class surface:
    def __init__(self):
        # best way to init a member variable in a way that you can check if it is valid
        self.verts = None
        self.faces = None

def demoSurfaceFromNRRD():
    import nrrd
    from skimage import measure

    img, header = nrrd.read('C:\\Users\\noblejh\\Box Sync\\EECE_395\\0522c0001\\img.nrrd')
    s = surface()
    s.verts, s.faces, _, _ = measure.marching_cubes(img, level=700)

    win = vtkWin()
    win.addSurf(s.verts, s.faces, color=[1,.9,.8])
    win.start()

    voxsz = [header['space directions'][0][0],header['space directions'][1][1],header['space directions'][2][2]]
    s.verts,s.faces,_,_ = measure.marching_cubes(img,level=700, spacing=voxsz)

    win = vtkWin()
    win.addSurf(s.verts,s.faces,color=[1,.9,.8])
    win.start()



if __name__ == "__main__":

    demoPointsAndLines()
    # demoSurfaceAppearance()
    # demoSurfaceEdgesAndColors()
    # demoDepthOfField()
    # brainPointPick()
    # brainAnimation()
    # bouncingBallsAnimation()
    # demoSurfaceFromNRRD()
