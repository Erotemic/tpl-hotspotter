import os
os.chdir('hotspotter/tpl')
'''
build_index(self, pts, **kwargs) method of pyflann.index.FLANN instance
    This builds and internally stores an index to be used for
    future nearest neighbor matchings.  It erases any previously
    stored indexes, so use multiple instances of this class to
    work with multiple stored indices.  Use nn_index(...) to find
    the nearest neighbors in this index.
    
    pts is a 2d numpy array or matrix. All the computation is done
    in float32 type, but pts may be any type that is convertable
    to float32.
'''
import pyflann 
try:
    # Append the tpl lib to your path
    PYFLANN_DIR = os.path.dirname (os.path.realpath(pyflann.__file__))
    TPL_LIB_DIR = os.path.normpath(os.path.join(PYFLANN_DIR, 'lib', sys.platform))
    sys.path.append(TPL_LIB_DIR)
except Exception: 
    print('''You must download hotspotter\'s 3rd party libraries before you can
          run it.  git clone https://github.com/Erotemic:tpl-hotspotter.git
          tpl''')
import cPickle
import numpy as np

#alpha = xrange(0,128)
#pts  = np.random.dirichlet(alpha,size=10000, dtype=np.uint8)
#qpts = np.random.dirichlet(alpha,size=100, dtype=np.uint8)

# Test parameters
nump = 10000
numq = 100
dims = 128
dtype = np.float32
# Create query and database data
pts  = array(np.random.randint(0,255,(nump,dims)), dtype=dtype)
qpts = array(np.random.randint(0,255,(nump,dims)), dtype=dtype)
# Create flann object
flann = pyflann.FLANN()
# Build kd-tree index over the data
build_params = flann.build_index(pts)
# Find the closest few points to num_neighbors
rindex, rdist = flann.nn_index(qpts, num_neighbors=3)

# Save the data to disk
np.savez('ptsdata.npz', pts)
npload_pts = np.load('ptsdata.npz')
pts2 = npload_pts['arr_0']

flann.save_index('index.flann')
flann.delete_index()

flann2 = pyflann.FLANN()
flann2.load_index('index.flann',pts2)
rindex, rdist = flann2.nn_index(qpts, num_neighbors=3)

