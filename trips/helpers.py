#!/usr/bin/env python
"""
Definition of test problems
--------------------------------------------------------------------------
Created December 10, 2022 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha and Connor Sanderford"
__copyright__ = "Copyright 2022, TRIPs-Py library"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Mirjeta Pasha and Connor Sanderford"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com and csanderf@asu.edu; connorsanderford@gmail.com"

from trips.io import *
from trips.operators import *
from trips.solvers.AnisoTV import *
from matplotlib import transforms
from scipy import ndimage
import matplotlib.pyplot as plt
# from trips.cil_io import *
from trips.testProblems import *
from trips.utils import *

def plot_singular_vectors_svd(Operator, size, parameter = 'A'):

    if parameter == 'A':
        U, S, V = np.linalg.svd(Operator)
        V = V.T
        n = 2
        m = 6
        fig, axes = plt.subplots(n, m)
        v_i = [V[:, k] for k in range(m)]
        v_i += [V[:, k] for k in range(130, 130+m)]
        right_singular = np.array(v_i)
        immax = np.max(right_singular)
        immin = np.min(right_singular)
        k = 0
        for i in range(n):
            for j in range(m):
                image = axes[i][j].imshow(-V[:, k].reshape((size, size)), vmin=immin, vmax=immax, cmap='inferno')
                axes[i][j].axis('off')
                axes[i][j].set_title(r'$v_{' + str(k) + r'}$')
                k += 1
            k = 50
        plt.subplots_adjust(bottom=0, top=0.7, left = 0, right=1)
        fig.colorbar(image, ax=axes.ravel().tolist())
        plt.savefig('v_vectors.png', bbox_inches='tight')
        plt.show()
    else:
        n = 2
        m = 6
        fig, axes = plt.subplots(n, m)
        v_i = [V[:, k] for k in range(m)]
        v_i += [V[:, k] for k in range(130, 130+m)]
        right_singular = np.array(v_i)
        immax = np.max(right_singular)
        immin = np.min(right_singular)
        k = 0
        for i in range(n):
            for j in range(m):
                image = axes[i][j].imshow(-V[:, k].reshape((size, size)), vmin=immin, vmax=immax, cmap='inferno')
                axes[i][j].axis('off')
                axes[i][j].set_title(r'$v_{' + str(k) + r'}$')
                k += 1
            k = 50
        plt.subplots_adjust(bottom=0, top=0.7, left = 0, right=1)
        fig.colorbar(image, ax=axes.ravel().tolist())
        plt.savefig('v_vectors.png', bbox_inches='tight')
        plt.show()

def plot_singular_values_svd(Operator, parameter = 'A'):
    A = check_operator_type(Operator)
    if parameter == 'A':
        U, S, V = np.linalg.svd(Operator)
        plt.plot(S)
        plt.title('Singular values of $A$')
        plt.xlabel('$\ell$')
        plt.ylabel('$\sigma_{\ell}$')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
        plt.savefig('singular_values.png', bbox_inches='tight')
        plt.show()
    else:
        plt.plot(S)
        plt.title('Singular values of $A$')
        plt.xlabel('$\ell$')
        plt.ylabel('$\sigma_{\ell}$')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
        plt.savefig('singular_values.png', bbox_inches='tight')
        plt.show()

def check_imagesize_toreshape(existingimage, chooseimage, old_size, newsize):
    path_package = '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package'
    if (old_size[0] != newsize[0] or old_size[1] != newsize[1]):
        Deblur.plot_rec(existingimage.reshape((shape), order = 'F'), save_imgs = False)
        temp_im = Image.open(path_package + '/demos/data/images/'+chooseimage+'_'+str(newsize[0])+'.jpg')
        image_new =  np.array(temp_im.resize((newsize[0], newsize[1])))
        spio.savemat(path_package + '/demos/data/images/'+chooseimage+'_'+str(newsize[0])+'.mat', mdict={'x_true': image_new})
    return image_new

def plot_recstructions_series(img, shape, dynamic, testproblem, geome_x, geome_x_small, save_imgs= True, save_path='./reconstruction/Emoji'):
    """
    Plot the reconstruction.
    """
    [nx, ny, nt] = shape
    if testproblem == 'gelPhantom':
        if dynamic == True:
            titles_sinos = ["Time-frame {}".format(i) for i in [0,5,10,16]]
            show2D(img.reshape(geome_x.shape), slice_list = [0,5,10,16], num_cols=4, origin="upper",fix_range=(0,0.065),
                       cmap="inferno", title=titles_sinos, size=(25, 20))
        else:
            for i in range(nt):
                rows = int(np.sqrt(nt))+1
                columns = int(np.sqrt(nt))+1
                fig = plt.figure(figsize=(50, 50))  
                fig.add_subplot(rows, columns, i+1)
                titles_sinos = ["Time-frame {}".format(i)]
                show2D(img[i].reshape(geome_x_small.shape, order = 'F' ), slice_list = [i], num_cols=4, origin="upper",fix_range=(0,0.065),
                   cmap="inferno", title=titles_sinos, size=(25, 20))
            
    else:
        if np.ndim(img)!=3: img = img.reshape((nx, ny, nt),order='F')
        # plot
        import matplotlib.pyplot as plt
        plt.set_cmap('inferno')
        if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
        rows = int(np.sqrt(nt))+1
        columns = int(np.sqrt(nt))+1
        fig = plt.figure(figsize=(50, 50))   
        if dynamic:
            for i in range(nt):
                fig.add_subplot(rows, columns, i+1)
                aa = img[:,:,i]
                rotated_img = ndimage.rotate(aa,-90, mode='constant')
                plt.imshow(rotated_img)
    #             plt.title('t = '+str(i),fontsize=16)
                if save_imgs: 
                    plt.savefig(save_path+'/emoji_dynamic_'+str(i).zfill(len(str(img.shape[2])))+'.png',bbox_inches='tight')
                    plt.pause(.1)
                    plt.draw()
        else:
            rows = int(np.sqrt(nt))+1
            columns = int(np.sqrt(nt))+1
            fig = plt.figure(figsize=(50, 50)) 
            for i in range(nt):
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(img[i].reshape((nx, ny)))
    #             plt.title('t = '+str(i),fontsize=16)
                if save_imgs: 
                    for i in range(nt):
                        plt.savefig(save_path+'/emoji_static_'+'.png',bbox_inches='tight')
                        plt.pause(.1)
                        plt.draw()


def plot_sinograms(B, nt, sinoshape, save_imgs= False):
    """
    Plot the sinograms.
    """
    plt.set_cmap('inferno')
    if save_imgs and not os.path.exists(save_path): os.makedirs(save_path)
    f = plt.figure(figsize=(15,15))
    f, axarr = plt.subplots(6,6)
    count = 0
    for i in range(6):
        for j in range(6):
            count = count + 1
            if count >32:
                break
            axarr[i,j].imshow(B[count-1].reshape(217, 10, order='F'), aspect = 'auto')
