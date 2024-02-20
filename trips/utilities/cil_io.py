#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, and Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, and Arizona State University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; csanderf@asu.edu; connorsanderford@gmail.com;"

import requests
from scipy import sparse
import scipy.io as spio
import numpy as np
import h5py

import os
from os import mkdir
from os.path import exists

import astra

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


def get_stempo_data(data_set = 'real', data_thinning = '2'):
        """
        Generate stempo observations
        """
        data_file = {'simulation':'stempo_ground_truth_2d_b4','real':'stempo_seq8x45_2d_b'+str(data_thinning)}[data_set]+'.mat'
        if not os.path.exists('./data/stempo_data'): os.makedirs('./data/stempo_data')
        if not os.path.exists('./data/stempo_data/'+data_file):
            import requests
            print("downloading...")
            r = requests.get('https://zenodo.org/record/7147139/files/'+data_file)
            with open('./data/'+data_file, "wb") as file:
                file.write(r.content)
            print("Stempo data downloaded.")
        if  data_set=='simulation':
            truth = spio.loadmat('./data/stempo_data/'+data_file)
            image = truth['obj']
            nx, ny, nt = 560, 560, 20
            anglecount = 10
            rowshift = 5
            columnsshift = 14
            nt = 20
            angleVector = list(range(nt))
            for t in range(nt):
                angleVector[t] = np.linspace(rowshift*t, 14*anglecount+ rowshift*t, num = anglecount+1)
            angleVectorRad = np.deg2rad(angleVector)
                    # Generate matrix versions of the operators and a large bidiagonal sparse matrix
            N = nx         # object size N-by-N pixels
            p = int(np.sqrt(2)*N)    # number of detector pixels
            # view angles
            theta = angleVectorRad#[0]#np.linspace(0, 2*np.pi, q, endpoint=False)   # in rad
            q = theta.shape[1]          # number of projection angles
            source_origin = 3*N                     # source origin distance [cm]
            detector_origin = N                       # origin detector distance [cm]
            detector_pixel_size = (source_origin + detector_origin)/source_origin
            detector_length = detector_pixel_size*p 
            saveA = list(range(nt))
            saveb = np.zeros((p*q, nt))
            saveb_true = np.zeros((p*q, nt))
            savee = np.zeros((p*q, nt))
            savedelta = np.zeros((nt, 1))
            savex_true = np.zeros((nx*ny, nt))
            B = list(range(nt))
            count = np.int_(360/nt)
            for i in range(nt):
                proj_geom = astra.create_proj_geom('fanflat', detector_pixel_size, p, theta[i], source_origin, detector_origin)
                vol_geom = astra.create_vol_geom(N, N)
                proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
                mat_id = astra.projector.matrix(proj_id)
                A_n = astra.matrix.get(mat_id)
                x_true = image[:, :, count*i]
                x_truef_sino = x_true.flatten(order='F') 
                savex_true[:, i] = x_truef_sino
                sn = A_n*x_truef_sino
                b_i = sn.flatten(order='F') 
                sigma = 0.01 # noise level
                e = np.random.normal(0, 1, b_i.shape[0])
                e = e/np.linalg.norm(e)*np.linalg.norm(b_i)*sigma
                delta = np.linalg.norm(e)
                b_m = b_i + e
                saveA[i] = A_n
                B[i] = b_m
                saveb_true[:, i] = sn
                saveb[:, i] = b_m
                savee[:, i] = e
                savedelta[i] = delta
                astra.projector.delete(proj_id)
                astra.matrix.delete(mat_id)
            A = sps.block_diag((saveA))    
            b = saveb.flatten(order ='F') 
            # xf = savex_true.flatten(order = 'F')
            truth = savex_true.reshape((nx, ny, nt), order='F').transpose((2,0,1))
        elif data_set=='real':
            import h5py
            N = int(2240/data_thinning) # 140
            nx, ny, nt =  N, N, 8
            N_det = N
            N_theta = 45
            theta = np.linspace(0,360,N_theta,endpoint=False)
            # Load measurement data as sinogram
            # data = spio.loadmat('./data/'+data_file) # scipy.io does not support Matlab struct
            data = h5py.File('./data/stempo_data/'+data_file,'r')
            CtData = data["CtData"]
            m = np.array(CtData["sinogram"]).T # strange: why is it transposed?
            # Load parameters
            param = CtData["parameters"]
            f = h5py.File('A_seqData.mat') # it is created by `create_ct_matrix_2d_fan_astra.m` separately
            fA = f["A"]
            # Extract information
            Adata = np.array(fA["data"])
            Arowind = np.array(fA["ir"])
            Acolind = np.array(fA["jc"])
            # Need to know matrix size (shape) somehow
            n_rows = N_det*N_theta # 6300
            n_cols = N*N
            Aloaded = sps.csc_matrix((Adata, Arowind, Acolind), shape=(n_rows, n_cols))
            saveA = list(range(nt))
            saveb = np.zeros((n_rows, nt))
            savee = np.zeros((n_rows, nt))
            savedelta = np.zeros((nt, 1))
            B = list(range(nt))
            for i in range(nt):
                tmp = m[45*(i):45*(i+1), :]
                b_i = tmp.flatten()
                sigma = 0.01 # noise level
                e = np.random.normal(0, 1, b_i.shape[0])
                e = e/np.linalg.norm(e)*np.linalg.norm(b_i)*sigma
                delta = np.linalg.norm(e)
                b_m = b_i + e
                saveA[i] = Aloaded
                B[i] = b_m
                saveb[:, i] = b_m
                savee[:, i] = e
                savedelta[i] = delta
            A = sps.block_diag((saveA))    
            b = saveb.flatten(order ='F')
            truth = None
        return A, b, saveA, B, nx, ny, nt, savedelta, truth