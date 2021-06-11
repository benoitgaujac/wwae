import os
import re
import pdb

from math import sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utils

import typing


def save_train(opts: dict, data_train: np.ndarray, data_test: np.ndarray,
                     rec_train, rec_test,
                     samples,
                     loss: list, loss_test: list,
                     losses: list, losses_test: list,
                     mse: list, mse_test: list,
                     fid_rec: list, fid_gen: list,
                     exp_dir: str,
                     filename: str) -> None:

    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img5 | img6

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   obj curves
        img5    -   split loss curves
        img6    -   mse/fid

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = data_train.shape[-1] == 1

    if opts['input_normalize_sym']:
        data_train = data_train / 2. + 0.5
        data_test = data_test / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        rec_test = rec_test / 2. + 0.5
        samples = samples / 2. + 0.5

    images = []
    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test[:num_pics])]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        assert len(sample) == num_pics
        assert len(sample) == len(recon)
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2
        # Figuring out a layout
        image = np.split(merged[:num_pics], num_cols)
        image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
        image = np.concatenate(image, axis=2)
        image = np.split(image,image.shape[0],axis=0)
        image = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
        image = np.concatenate(image, axis=1)
        image = image[0]
        if greyscale:
            image = 1. - image
        images.append(image)

    ### Sample plots
    assert len(samples) == num_pics
    # Figuring out a layout
    image = np.split(samples, num_cols)
    image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=2)
    image = np.split(image,image.shape[0],axis=0)
    image = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=1)
    image = image[0]
    if greyscale:
        image = 1. - image
    images.append(image)

    img1, img2, img3 = images

    # Creating a pyplot fig
    dpi = 100
    height_pic = img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = 4 * 2*height_pic / float(dpi)
    fig_width = 6 * 2*width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(2, 3)

    # Filling in separate parts of the plot

    # First samples and reconstructions
    for img, (gi, gj, title) in zip([img1, img2, img3],
                             [(0, 0, 'Train reconstruction'),
                              (0, 1, 'Test reconstruction'),
                              (0, 2, 'Generated samples')]):
        plt.subplot(gs[gi, gj])
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            ax = plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            ax = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        ax = plt.subplot(gs[gi, gj])
        plt.text(0.47, 1., title,
                 ha="center", va="bottom", size=20, transform=ax.transAxes)
        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)

    ### The loss curves
    ax = plt.subplot(gs[1, 0])
    # train
    array_losses = np.array(losses).reshape((-1,5))
    rec = array_losses[:,0]
    reg = opts['beta']*array_losses[:,1]
    for y, (color, label) in zip([loss, rec, reg],
                                [('black', 'loss tr'),
                                ('blue', 'rec tr'),
                                ('red', 'reg tr')]):
        if label!='reg tr' or opts['beta']>0.:
            total_num = len(y)
            # x_step = max(int(total_num / 200), 1)
            # x = np.arange(1, len(y) + 1, x_step)
            # y = np.log(y[::x_step])
            x = np.arange(1, total_num + 1)
            y = np.log(y)
            plt.plot(x, y, linewidth=2, color=color, linestyle='--', label=label)
    # test
    array_losses = np.array(losses_test).reshape((-1,5))
    rec = array_losses[:,0]
    reg = opts['beta']*array_losses[:,1]
    for y, (color, label) in zip([loss_test, rec, reg],
                                [('black', 'loss te'),
                                ('blue', 'rec te'),
                                ('red', 'reg te')]):
        if label!='reg te' or opts['beta']>0.:
            total_num = len(y)
            # x_step = max(int(total_num / 200), 1)
            # x = np.arange(1, len(y) + 1, x_step)
            # y = np.log(y[::x_step])
            x = np.arange(1, total_num + 1)
            y = np.log(y)
            plt.plot(x, y, linewidth=4, color=color, label=label)

    plt.grid(axis='y')
    handles, labels = plt.gca().get_legend_handles_labels()
    order, nlabels = [], len(labels)
    for i in range(nlabels):
        if i%2==0:
            order.append(int((nlabels+i)/2))
        else:
            order.append(int(i/2))
    # order = [3,0,4,1,5,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')
    # plt.legend(loc='upper right')
    plt.text(0.47, 1., 'Log Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### The split loss curves
    ax = plt.subplot(gs[1, 1])
    # train
    array_losses = np.array(losses).reshape((-1,5))
    for y, (color, label) in zip([array_losses[:,2],
                                opts['gamma']*array_losses[:,3],
                                opts['beta']*array_losses[:,1],
                                array_losses[:,-1]],
                                [('blue', 'grd. cost tr'),
                                ('green', 'int. reg. tr'),
                                ('red', 'lat. reg. tr'),
                                ('magenta', 'cri. loss tr')]):
        # if (label!='lat. reg tr' or opts['beta']>0.) and (label!='int. reg. tr' or opts['gamma']>0.):
        if y[0]!=0. and y[-1]!=0.:
            total_num = len(y)
            # x_step = max(int(total_num / 200), 1)
            # x = np.arange(1, len(y) + 1, x_step)
            # y = np.log(y[::x_step])
            x = np.arange(1, total_num + 1)
            plt.plot(x, y, linewidth=1, color=color, linestyle='--', label=label)
    # test
    array_losses = np.array(losses_test).reshape((-1,5))
    for y, (color, label) in zip([array_losses[:,2],
                                opts['gamma']*array_losses[:,3],
                                opts['beta']*array_losses[:,1],
                                array_losses[:,-1]],
                                [('blue', 'grd. cost te'),
                                ('green', 'int. reg. te'),
                                ('red', 'lat. reg te'),
                                ('magenta', 'cri. loss tr')]):
        # if (label!='lat. reg tr' or opts['beta']>0.) and (label!='int. reg. tr' or opts['gamma']>0.):
        if y[0]!=0. and y[-1]!=0.:
            total_num = len(y)
            # x_step = max(int(total_num / 200), 1)
            # x = np.arange(1, len(y) + 1, x_step)
            # y = np.log(y[::x_step])
            x = np.arange(1, total_num + 1)
            plt.plot(x, y, linewidth=3, color=color, label=label)
    plt.grid(axis='y')
    handles, labels = plt.gca().get_legend_handles_labels()
    order, nlabels = [], len(labels)
    for i in range(nlabels):
        if i%2==0:
            order.append(int((nlabels+i)/2))
        else:
            order.append(int(i/2))
    # order = [3,0,4,1,5,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')
    # plt.legend(loc='upper right')
    plt.text(0.47, 1., 'split Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)
    ### The mse/fid curves
    base = plt.cm.get_cmap('tab10')
    color_list = base(np.linspace(0, 1, 6))
    ax = plt.subplot(gs[1, 2])
    # Test
    y = mse_test
    # y = y[::x_step]
    plt.plot(x, y, linewidth=4, color=color_list[2], label='MSE test')
    # Train
    y = mse
    # y = y[::x_step]
    plt.plot(x, y, linewidth=2, color=color_list[2], linestyle='--', label='MSE')
    if len(fid_rec)>0:
        # FID rec
        y = fid_rec
        # y = y[::x_step]
        plt.plot(x, y, linewidth=4, color=color_list[0], label='FID rec')
        # FID rec
        y = fid_gen
        # y = y[::x_step]
        plt.plot(x, y, linewidth=2, color=color_list[1], label='FID gen')

    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.text(0.47, 1., 'MSE/FID curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### Saving plots
    # Plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()


def plot_critic_pretrain_loss(opts, loss, exp_dir,filename):
    fig, ax = plt.subplots()
    total_num = len(loss)
    x = np.arange(total_num)
    ax.plot(x, loss, linewidth=0.7)
    ax.grid(True, which='major', axis='y')
    xticks = x[::int(total_num/10)]
    xlabels = x[::int(total_num/10)]*opts['pretrain_critic_nit']/200
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels.astype(int))
    ax.set_xlabel('Pretraining iterations')
    ax.set_ylabel('Loss')
    # ax.legend(loc='lower right')
    ax.set_title('Critic pretraining loss')

    ### Saving plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),cformat='png')
    plt.close()


def plot_embedded_shift(opts: dict, encoded: np.ndarray, exp_dir: str) -> None:
    nobs, nshift = np.shape(encoded)[:2]
    codes = encoded.reshape([nobs*nshift,-1])
    labels = np.repeat(np.arange(nobs),nshift)
    if np.shape(codes)[-1]==2:
        embedding = codes
    else:
        if opts['embedding']=='pca':
            embedding = PCA(n_components=2).fit_transform(codes)
        elif opts['embedding']=='umap':
            embedding = umap.UMAP(n_neighbors=15,
                                    min_dist=0.2,
                                    metric='correlation').fit_transform(codes)
        else:
            assert False, 'Unknown %s method for embedgins vizu' % opts['embedding']
    # Creating a pyplot fig
    dpi = 100
    height_pic = 500
    width_pic = 500
    fig_height =  height_pic / float(dpi)
    fig_width =  width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.8, linewidths=0.,
                c=labels, s=40, cmap=discrete_cmap(10, base_cmap='tab10'))
    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.05
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify
    ymin = np.amin(embedding[:,1])
    ymax = np.amax(embedding[:,1])
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # plt.legend(loc='best')
    # plt.text(0.47, 1., 'UMAP vizu', ha="center", va="bottom",
    #                                         size=20, transform=ax.transAxes)
    plt.title(opts['embedding'] + ' vizualisation of the shifted latent codes')
    # Removing ticks if needed
    if opts['embedding']=='umap':
        # plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False)
        plt.xticks([])
        plt.yticks([])
    ### Saving plot
    save_path = os.path.join(exp_dir,'test_plots')
    utils.create_dir(save_path)
    filename = opts['cost'] + '_embedded_shifted.png'
    fig.savefig(utils.o_gfile((save_path, filename),'wb'),
                dpi=dpi, cformat='png',bbox_inches='tight',pad_inches=0.05)
    plt.close()


def plot_interpolation(opts: dict, interpolations: np.ndarray, exp_dir: str, filename: str, train=True) -> None:
    ### Reshaping images
    greyscale = interpolations.shape[-1] == 1
    if opts['input_normalize_sym']:
        interpolations = interpolations / 2. + 0.5
    white_pix = 4
    num_rows = np.shape(interpolations)[0]
    num_cols = np.shape(interpolations)[1]
    # images = np.concatenate(np.split(interpolations,num_cols,axis=1),axis=3)
    # images = images[:,0]
    # images = np.concatenate(np.split(images,num_rows,axis=0),axis=1)
    # images = images[0]
    # if greyscale:
    #     images = 1. - images
    images = np.split(interpolations,num_cols,axis=1)
    images = [np.pad(img, ((0,0),(0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in images]
    images = np.concatenate(images,axis=3)
    images = images[:,0]
    images = np.split(images,num_rows,axis=0)
    images = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in images]
    images = np.concatenate(images,axis=1)
    images = images[0]
    if greyscale:
        images = 1. - images
    ### Creating plot
    dpi = 100
    height_pic = images.shape[0]
    width_pic = images.shape[1]
    fig_height = 10*height_pic / float(dpi)
    fig_width = 10*width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    if greyscale:
        images = images[:, :, 0]
        # in Greys higher values correspond to darker colors
        plt.imshow(images, interpolation='none', vmin=0., vmax=1., cmap='Greys')
    else:
        plt.imshow(images, interpolation='none', vmin=0., vmax=1.)
    # Removing axes, ticks
    plt.axis('off')
    # # placing subplot
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    ### Saving plot
    if train:
        plots_dir = 'train_plots'
    else:
        plots_dir = 'test_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),dpi=dpi,cformat='png')
    plt.close()


def save_test(opts: dict, data: np.ndarray, reconstructions, samples, encoded, labels: typing.Optional[str]=None, exp_dir: typing.Optional[str]=None) -> None:

    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = data.shape[-1] == 1

    if opts['input_normalize_sym']:
        data = data / 2. + 0.5
        reconstructions = reconstructions / 2. + 0.5
        samples = samples / 2. + 0.5

    ### Reconstruction plots
    # Arrange pics and reconstructions in a proper way
    assert len(data) == num_pics
    assert len(data) == len(reconstructions)
    pics: list = []
    merged = np.vstack([reconstructions, data])
    r_ptr = 0
    w_ptr = 0
    for _ in range(int(num_pics / 2)):
        merged[w_ptr] = data[r_ptr]
        merged[w_ptr + 1] = reconstructions[r_ptr]
        r_ptr += 1
        w_ptr += 2
    # Figuring out a layout
    image = np.split(merged[:num_pics], num_cols)
    image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=2)
    image = np.split(image,image.shape[0],axis=0)
    image = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=1)
    rec = image[0]
    if greyscale:
        rec = 1. - rec

    ### Samples plots
    # Figuring out a layout
    image = np.split(samples, num_cols)
    image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=2)
    image = np.split(image,image.shape[0],axis=0)
    image = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=1)
    gen = image[0]
    if greyscale:
        gen = 1. - gen

    ### Embedding plots
    if np.shape(encoded)[-1]>1:
        if np.shape(encoded)[-1]==3:
            embedding = encoded
        else:
            if opts['embedding']=='pca':
                embedding = PCA(n_components=2).fit_transform(encoded)
            elif opts['embedding']=='umap':
                embedding = umap.UMAP(n_neighbors=15,
                                        min_dist=0.2,
                                        metric='correlation').fit_transform(encoded)
            else:
                assert False, 'Unknown %s method for embedgins vizu' % opts['embedding']
        # Creating a pyplot fig
        dpi = 100
        height_pic = 600
        width_pic = 600
        colors = {0:'b', 1:'r', 2:'c', 3:'m'}
        if opts['dataset'] == 'shifted_mnist':
            labels_names = {0:'top-right 0', 1:'top-right 1', 2:'bottom-left 0', 3:'bottom-left 1'}
        elif opts['dataset'] == 'shifted_3pos_mnist':
            labels_names = {1:'top-right', 2:'middle', 3:'bottom-left'}
        elif opts['dataset'] == 'rotated_mnist':
            # labels_names = {0:'original 1', 1:'original 5', 2:'rotated 1', 3:'rotated 5'}
            labels_names = {1:'1', 3:'3', 7:'7', 8:'8'}
        elif opts['dataset'] == 'gmm':
            labels_names = {0:'top-left', 1:'bottom-right'}
        else:
            assert False, 'Unknown {} dataset'.format(opts['dataset'])
        fig_height =  height_pic / float(dpi)
        fig_width =  width_pic / float(dpi)
        fig = plt.figure(figsize=(fig_width, fig_height))
        if labels is not None:
            classes = np.unique(labels)
            classes = [classes[i] for i in range(len(classes))]
            for c in classes:
                idx = np.where(labels==c)
                plt.scatter(embedding[idx, 0], embedding[idx, 1], alpha=0.8, linewidths=0.,
                            # c=colors[c],
                            label=labels_names[c] , s=40)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.8, linewidths=0., s=40)
        xmin = np.amin(embedding[:,0])
        xmax = np.amax(embedding[:,0])
        magnify = 0.05
        width = abs(xmax - xmin)
        xmin = xmin - width * magnify
        xmax = xmax + width * magnify
        ymin = np.amin(embedding[:,1])
        ymax = np.amax(embedding[:,1])
        width = abs(ymin - ymax)
        ymin = ymin - width * magnify
        ymax = ymax + width * magnify
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        if labels is not None:
            plt.legend(loc='best')
        # plt.text(0.47, 1., 'UMAP vizu', ha="center", va="bottom",
        #                                         size=20, transform=ax.transAxes)
        if np.shape(encoded)[-1]==2:
            plt.title('latent codes')
        else:
            plt.title(opts['embedding'] + ' vizualisation of latent codes')
        # Removing ticks if needed
        if opts['embedding']=='umap':
            # plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False)
            plt.xticks([])
            plt.yticks([])
        ### Saving plot
        save_path = os.path.join(exp_dir,'test_plots')
        utils.create_dir(save_path)
        filename = opts['cost'] + '_embedded.png'
        fig.savefig(utils.o_gfile((save_path, filename),'wb'),
                    dpi=dpi, cformat='png',bbox_inches='tight',pad_inches=0.05)
        plt.close()


    ### Creating plot
    for img, title in zip([rec, gen],
                        ['reconstructions.png',
                        'samples.png']):
        dpi = 100
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        fig_height = 10*height_pic / float(dpi)
        fig_width = 10*width_pic / float(dpi)
        fig = plt.figure(figsize=(fig_width, fig_height))
        if greyscale:
            img = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(img, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        ### Saving plot
        save_path = os.path.join(exp_dir,'test_plots')
        utils.create_dir(save_path)
        fig.savefig(utils.o_gfile((save_path, title), 'wb'),dpi=dpi,cformat='png')
        plt.close()


def plot_cost_shift(rec_sr: np.ndarray, mse_sr: np.ndarray, ground_os: np.ndarray, mse_os: np.ndarray, exp_dir: str) -> None:
        nshift = len(rec_sr)
        dpi = 100
        fig_height = 500 / float(dpi)
        fig_width = 2*500 / float(dpi)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height))
        # obs vs shifted
        for y, (label,color) in zip([ground_os, mse_os],
                            [('ground','b'),
                            ('MSE','r')]):
            axes[0].plot(list(range(nshift)),y,label=label,color=color,)
        # add linear reg for ground cost
        deg = 1
        npoints_to_fit = 3
        x = np.arange(npoints_to_fit)
        p = np.polyfit(x,ground_os[:npoints_to_fit],deg)
        npoints_to_plot = max(int(len(ground_os)/4),3)
        x = np.arange(npoints_to_plot)
        reg = np.zeros(npoints_to_plot)
        for n in range(deg+1):
            reg += p[n]*x**(deg-n)
        axes[0].plot(x, reg, c='y', linestyle='--', alpha=0.8)

        xticks = np.arange(nshift)
        axes[0].set_xticks(xticks[::4])
        axes[0].set_xticklabels(2*xticks[::4])
        axes[0].set_xlabel('pixels shifted')
        axes[0].grid(True, which='major', axis='y')
        axes[0].legend(loc='best')
        axes[0].set_ylabel('cost(obs,shifted)')
        axes[0].set_title('cost vs pixel shift')
        # shift vs reconstructions
        for y, (label,color) in zip([rec_sr, mse_sr],
                            [('rec','b'),
                            ('MSE','r')]):
            axes[1].plot(list(range(nshift)),y,label=label,color=color,)
        # add linear reg for ground cost
        deg = 1
        npoints_to_fit = 3
        x = np.arange(npoints_to_fit)
        p = np.polyfit(x,rec_sr[:npoints_to_fit],deg)
        npoints_to_plot =  max(int(len(rec_sr)/4),3)
        x = np.arange(npoints_to_plot)
        reg = np.zeros(npoints_to_plot)
        for n in range(deg+1):
            reg += p[n]*x**(deg-n)
        axes[1].plot(x, reg, c='y', linestyle='--', alpha=0.8)
        xticks = np.arange(nshift)
        axes[1].set_xticks(xticks[::4])
        axes[1].set_xticklabels(2*xticks[::4])
        axes[1].set_xlabel('pixels shifted')
        axes[1].grid(True, which='major', axis='y')
        axes[1].legend(loc='best')
        axes[1].set_ylabel('cost(shifted,reconstruction)')
        axes[1].set_title('cost vs pixel shift')

        ### Saving plot
        save_path = os.path.join(exp_dir,'test_plots')
        utils.create_dir(save_path)
        fig.savefig(os.path.join(save_path, 'perurbation_stab.png'),
                    dpi=dpi,format='png', bbox_inches='tight')
        plt.close()


def plot_rec_shift(opts, shifted_obs: np.ndarray, shifted_rec: np.ndarray, exp_dir: str) -> None:

    greyscale = shifted_obs.shape[-1] == 1

    if opts['input_normalize_sym']:
        shifted_obs = shifted_obs / 2. + 0.5
        shifted_rec = shifted_rec / 2. + 0.5

    ncol = shifted_obs.shape[1]
    nrow = shifted_obs.shape[0]
    pics = []
    for r in range(2*nrow):
        if r%2==0:
            pics.append(shifted_obs[int(r/2)])
        else:
            pics.append(shifted_rec[int(r/2)])
    pics = np.array(pics)
    images = np.split(pics,ncol,axis=1)
    images = [np.pad(img, ((0,0),(0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in images]
    images = np.concatenate(images,axis=3)
    images = images[:,0]
    images = np.split(images,2*nrow,axis=0)
    images = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in images]
    images = np.concatenate(images,axis=1)
    img = images[0]
    if greyscale:
        img = 1.0 - img
    dpi = 100
    height_pic = img.shape[0]
    width_pic = img.shape[1]
    fig_height = 10*height_pic / float(dpi)
    fig_width = 10*width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    if greyscale:
        img = img[:, :, 0]
        # in Greys higher values correspond to darker colors
        plt.imshow(img, cmap='Greys',
                        interpolation='none', vmin=0., vmax=1.)
    else:
        plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
    # Removing axes, ticks
    plt.axis('off')
    # # placing subplot
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    ### Saving plot
    save_path = os.path.join(exp_dir,'test_plots')
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, 'reconstructions_shifted.png'), 'wb'),dpi=dpi,cformat='png')
    plt.close()


def discrete_cmap(N: int, base_cmap=None) -> typing.Any:
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)
