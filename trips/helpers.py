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

