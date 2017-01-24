#%%

import h5py
import os

os.chdir('/home/bbales2/modal/')

import vtkWriter

data = h5py.File("mytestfile.hdf5", "r")

print data.keys()

array = data['mydataset']

f = open('mydataset.vtk', 'w')
f.write(vtkWriter.toString(array))
f.close()

#for