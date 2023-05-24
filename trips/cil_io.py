import requests
from scipy import sparse
import scipy.io as spio
import numpy as np
import h5py

import os
from os import mkdir
from os.path import exists

from cil.framework import AcquisitionGeometry
from cil.plugins.astra.operators import ProjectionOperator
from cil.processors import Slicer

"""
from utilities_dynamic_ct
"""

# read all the 17 frames
def read_frames(file_path, file_name):
    
    mat_contents = spio.loadmat(os.path.join(file_path,file_name),\
                           mat_dtype = True,\
                           squeeze_me = True,\
                           struct_as_record = True) 
        
    # get type
    mat_type = mat_contents[file_name]['type']

    # get sinograms
    mat_sinograms = mat_contents[file_name]['sinogram']

    # get parameters
    parameters = mat_contents[file_name]['parameters']

    # extract Distance Source Detector
    distanceSourceDetector = parameters[0]['distanceSourceDetector'].item()

    # extract Distance Source Origin
    distanceSourceOrigin = parameters[0]['distanceSourceOrigin'].item()

    # extract geometric Magnification
    geometricMagnification = parameters[0]['geometricMagnification'].item()
    #or geometricMagnification = distanceSourceDetector/distanceSourceOrigin

    # angles in rad
    angles = parameters[0]['angles'].item() * (np.pi/180.) 

    # extract numDetectors
    numDetectors = int(parameters[0]['numDetectors'].item())

    # effective pixel size
    effectivePixelSize = parameters[0]['effectivePixelSize'].item()

    # effective pixel size
    pixelSizeRaw = parameters[0]['pixelSizeRaw'].item()
    pixelSize = parameters[0]['pixelSize'].item()

    # compute Distance Origin Detector
    distanceOriginDetector = distanceSourceDetector - distanceSourceOrigin
    distanceSourceOrigin = distanceSourceOrigin
    distanceOriginDetector = distanceOriginDetector  
    
    file_info = {}
    file_info['sinograms'] = mat_sinograms
    file_info['angles'] = angles
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['distanceSourceOrigin'] = distanceSourceOrigin
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['pixelSize'] = pixelSize
    file_info['pixelSizeRaw'] = pixelSizeRaw    
    file_info['effectivePixelSize'] = effectivePixelSize
    file_info['numDetectors'] = numDetectors
    file_info['geometricMagnification'] = geometricMagnification
    
    return file_info

# read extra frames: 1, 18
def read_extra_frames(file_path, file_name, frame):

    mat_contents = spio.loadmat(os.path.join(file_path,file_name),\
                               mat_dtype = True,\
                               squeeze_me = True,\
                               struct_as_record = True)

    # get type
    mat_type = mat_contents[frame]['type']

    # get sinograms
    mat_sinograms = mat_contents[frame]['sinogram'].item()

    # get parameters
    parameters = mat_contents[frame]['parameters']

    # extract Distance Source Detector
    distanceSourceDetector = parameters.item()['distanceSourceDetector']

    # extract Distance Source Origin
    distanceSourceOrigin = parameters.item()['distanceSourceOrigin']

    # extract geometric Magnification
    geometricMagnification = parameters.item()['geometricMagnification']

    # angles in rad
    angles = parameters.item()['angles'].item()

    # extract numDetectors
    numDetectors = int(parameters.item()['numDetectors'].item())

    # effective pixel size
    effectivePixelSize = parameters.item()['effectivePixelSize'].item()

    # effective pixel size
    pixelSizeRaw = parameters.item()['pixelSizeRaw'].item()
    pixelSize = parameters.item()['pixelSize'].item()

    # compute Distance Origin Detector
    distanceOriginDetector = distanceSourceDetector - distanceSourceOrigin
    distanceSourceOrigin = distanceSourceOrigin#/effectivePixelSize
    distanceOriginDetector = distanceOriginDetector#/effectivePixelSize
    
    file_info = {}
    file_info['sinograms'] = mat_sinograms
    file_info['angles'] = angles
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['distanceSourceOrigin'] = distanceSourceOrigin
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['pixelSize'] = pixelSize
    file_info['pixelSizeRaw'] = pixelSizeRaw    
    file_info['effectivePixelSize'] = effectivePixelSize
    file_info['numDetectors'] = numDetectors
    file_info['geometricMagnification'] = geometricMagnification
    
    return file_info


"""
gelphantom
"""

def get_gelPhantom():

    if not exists('./data'):
        mkdir("./data")

    if not exists('./data/gelphantom_data'):
        mkdir("./data/gelphantom_data")

    if exists(f"./data/gelphantom_data/GelPhantomData_b4.mat"):
        print('Data already downloaded.')
    else:
        r = requests.get(f'https://zenodo.org/record/3696817/files/GelPhantomData_b4.mat')
        with open(f'./data/gelphantom_data/GelPhantomData_b4.mat', "wb") as file:
            file.write(r.content)
        print("gelphantom data downloaded.")


def gen_gelPhantom():
    
    get_gelPhantom()

    path = os.path.abspath("./data/gelphantom_data")
    data_mat = "GelPhantomData_b4"
    file_info = read_frames(path, data_mat)
    # Get sinograms + metadata
    sinograms = file_info['sinograms']
    frames = sinograms.shape[0]
    angles = file_info['angles']
    distanceOriginDetector = file_info['distanceOriginDetector']
    distanceSourceOrigin = file_info['distanceSourceOrigin']
    # Correct the pixel size
    pixelSize = 2*file_info['pixelSize']
    numDetectors = file_info['numDetectors']
    ag = AcquisitionGeometry.create_Cone2D(source_position = [0, distanceSourceOrigin],
                                    detector_position = [0, -distanceOriginDetector])\
                                .set_panel(numDetectors, pixelSize)\
                                .set_channels(frames)\
                                .set_angles(angles, angle_unit="radian")\
                                .set_labels(['channel','angle', 'horizontal'])

    ig = ag.get_ImageGeometry()
    ig.voxel_num_x = 256
    ig.voxel_num_y = 256
    data = ag.allocate()
    for i in range(frames):
        data.fill(sinograms[i], channel = i) 
    step = 20;
    name_proj = "data_{}".format(int(360/step))
    data = Slicer(roi={'angle':(0,360,step)})(data)
    ag = data.geometry
    A = ProjectionOperator(ig, ag, 'cpu')
    n_t = frames
    n_x = ig.voxel_num_x
    n_y = ig.voxel_num_y
    # Generate the small data
    x0 = ig.allocate()
    x0_small = x0.get_slice(channel = 0)
    ag_small = data.get_slice(channel=0).geometry
    ig_small = x0_small.geometry
    A_small = ProjectionOperator(ig_small, ag_small, 'cpu')
    temp = A_small.direct(x0_small)
    AA = list(range(n_t))
    for ii in range(n_t):
        AA[ii] = A_small
    B = np.zeros((data.shape[1]*data.shape[2], frames))
    for i in range(frames):
        temp = ((data.array)[i, :, :]).flatten()
        B[:, i] = temp
    return A, data, AA, B, n_x, n_y, n_t, ig, ag, ig_small, ag_small