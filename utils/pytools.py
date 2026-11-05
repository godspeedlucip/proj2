import numpy as np
import os
import pydicom
import matplotlib
import json5
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio

def save_as_gif(seqs, file_name, min, max, duration=0.5, loop=0):

    seqs[seqs < min] = min
    seqs[seqs > max] = max
    seqs = np.round((seqs - min) / (max - min) * 255)

    imageio.mimsave(file_name, seqs, duration=duration, loop=loop)



def read_raw_data(file_name, w, h, data_type='float32', mode='hw'):

    file_temp = np.fromfile(file_name, dtype=data_type, sep="")
    slice = int(np.size(file_temp) / w / h)
    if mode == 'hw':
        file_temp = np.reshape(file_temp, [slice, h, w])
    elif mode == 'wh':
        file_temp = np.reshape(file_temp, [slice, w, h])
    else:
        print('Mode has not been implemented!')

    return slice, file_temp

def read_raw_data_all(dir, w=512, h=512, start_index=8, end_index=-4, data_type='float32'):

    file_list = os.listdir(dir)
    # file_lista = file_list[0]
    # print(file_lista[-17:-14])
    file_list.sort(key=lambda x: int(x[start_index: end_index]))
    slice = len(file_list)

    file_vol = np.zeros([slice, h, w], dtype=np.float32)

    for index in range(slice):

        file_temp = np.fromfile(dir + file_list[index], dtype=data_type, sep="").astype(np.float32)
        file_temp = file_temp.reshape([h, w])
        file_vol[index, :, :] = file_temp

    return file_vol

def dicomreader(filename):
    info = pydicom.read_file(filename)
    img = np.float32(info.pixel_array)
    return info, img

def listsorter(dir, strat_index, end_index):
    list = os.listdir(dir)
    # print(list)
    list.sort(key=lambda x: int(x[strat_index: end_index]))
    return list

def read_dicom_all(file_dir, sort_start, sort_end, w=512, h=512):

    file_names = listsorter(file_dir, strat_index=sort_start, end_index=sort_end)
    slice_number = len(file_names)
    volume = np.zeros([slice_number, w, h], dtype=np.float32)
    for index in range(slice_number):
        info, img = dicomreader(file_dir + file_names[index])
        # print(info.SliceThickness)
        # img = np.flipud(img)
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()
        volume[index, :, :] = img

    return info, volume

def read_json_data(file_dir):

    with open(file_dir, 'r') as file:
        data = json5.load(file)
    file.close()

    return data

def make_dirs(dir_path):

    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

def addPossionNoisy(CleanProj, I0=5e4):

    TempProj = I0 * np.exp(-CleanProj) * 20
    NoiseProj = np.random.poisson(TempProj) + 1
    NoiseProj = -np.log(NoiseProj/I0/20)

    return NoiseProj

def addPossionGaussionNoisy(Proj, Possion=1e7, Gaussion=[0, 10]):

    maxProj = np.max(Proj)
    Proj = Possion * np.exp(-Proj/maxProj)

    if Gaussion is not None:
        Proj = Proj + np.random.normal(loc=Gaussion[0], scale=Gaussion[1], size=np.shape(Proj))

    Proj = np.random.poisson(Proj) + 1
    Proj = -np.log(Proj/Possion)*maxProj

    return Proj

def addPossionGaussionNoisyV2(Proj, Possion=1e7, Gaussion=[0, 10]):

    Proj = np.exp(-Proj)
    maxProj = np.max(Proj)
    Proj = Proj / maxProj * Possion
    if Gaussion is not None:
        Proj = Proj + np.random.normal(loc=Gaussion[0], scale=Gaussion[1], size=np.shape(Proj))
    Proj = np.abs(Proj) + 1
    Proj = np.random.poisson(Proj) + 1
    Proj = -np.log(Proj/Possion*maxProj)

    return Proj

def genMask(imgX=512, imgY=512, imgZ=256, maskR=230):

    maskSlice = np.zeros([imgX, imgY, 1], dtype=np.float32)

    for indexX in range(imgX):
        for indexY in range(imgY):
            if (indexX - imgX/2)**2 + (indexY - imgY/2)**2 <= maskR**2:
                maskSlice[indexX, indexY, :] = 1

    maskVol = np.tile(maskSlice, imgZ).transpose()
    # print(maskVol.shape)
    return maskVol

def fill_zeros(arr):
    rows, cols = arr.shape
    arr_corr = np.zeros_like(arr)
    zero_indices = np.argwhere(arr == 0)

    for index in zero_indices:
        row, col = index
        neighbors = []
        for i in range(max(0, row - 1), min(rows, row + 2)):
            for j in range(max(0, col - 1), min(cols, col + 2)):
                if (i, j) != (row, col):
                    neighbors.append(arr[i, j])

        arr_corr[row, col] = np.mean(neighbors)

    return arr_corr

def remove_zeros(sgm):
    sgm_corr = np.zeros_like(sgm)
    for i in range(sgm.shape[0]):
        sgm_corr[i, :, :] = fill_zeros(sgm[i, :, :])
    return sgm_corr


def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalize CDF to the range [0, 1]
    return cdf_normalized


def hist_match(source, template, bins=5):
    source_flat = source.flatten()
    template_flat = template.flatten()

    # Calculate histogram for source and template
    s_hist, s_bins = np.histogram(source_flat, bins=bins)
    t_hist, t_bins = np.histogram(template_flat, bins=bins)

    # Calculate CDF for source and template
    s_cdf = calculate_cdf(s_hist)
    t_cdf = calculate_cdf(t_hist)

    # Create a mapping from source CDF to template CDF
    interp_values = np.interp(s_cdf, t_cdf, t_bins[:-1])

    # Map source image pixels to template
    matched_flat = np.interp(source_flat, s_bins[:-1], interp_values)

    # Reshape to original shape
    matched = matched_flat.reshape(source.shape)

    return matched

class plotImg3D:  # noqa: N801
    """
    plotImg3D(cube, dim)
        plots figure
    default: progressive in slices following
        axis (dim)

    # plotImg3D plots the image slice by slice.

    # List of optional parameters:

    # 'Dim': specifies the dimension for plotting.
    #    Dim can be 'X','Y','Z'
    #    In python, X for the last dim and Z for the first dim

    dimension = "Z"

    # 'Step': step size of the plotting. Useful when images are big or one just
    # wants an overview of the result

    step = 2
    # 'Colormap': Defines the colormap used to plot. Default is 'gray'.

    colormap = "plasma"
    colormap = "magma"
    colormap = "gray"
    colormap = "viridis"

    # 'Clims': Defines the data limits for the color, usefull when one wants to
    # see some specific range. The default computes the 5% and 95% percentiles
    # of the data and uses that as limit.

    clims = [0, 1]

    # 'Savegif': allows to save the plotted figure as an animated gif,
    # specified by the given filename.

    giffilename = "demo5image.gif"

    # 'Slice': allows to plot a single slice .Will overwrite the behaviour
    # of 'Step'
    slice = 64

    # Lets go for it

    plotImg3D(imgFDK, dim=dimension, step=step, clims=clims, colormap=colormap, savegif=giffilename)

    """

    def __init__(
        self,
        cube,
        dim=None,
        slice=None,
        step=1,
        savegif=None,
        colormap="gray",
        clims=None,
        show_plot=None,
    ):
        self.cube = cube
        self.dim = dim
        self.slice = slice
        self.dimint = None  # keeps track of what dim
        self.dimlist = ["X", "Y", "Z", "x", "y", "z", None]  # accepted parameters for dim
        self.step = step
        self.savegif = savegif
        self.colormap = colormap
        if clims is None:
            self.min_val = np.amin(self.cube)
            self.max_val = np.amax(self.cube)
        else:
            self.min_val = clims[0]
            self.max_val = clims[1]
        if show_plot is None:
            # https://matplotlib.org/stable/tutorials/introductory/usage.html#backends
            backend = matplotlib.get_backend()
            if backend in [
                "GTK3Agg",
                "GTK3Cairo",
                "MacOSX",
                "nbAgg",
                "Qt4Agg",
                "Qt4Cairo",
                "Qt5Agg",
                "Qt5Cairo",
                "TkAgg",
                "TkCairo",
                "WebAgg",
                "WX",
                "WXAgg",
                "WXCairo",
                "module://ipykernel.pylab.backend_inline",
            ]:
                self.show_plot = True
            elif backend in ["agg", "cairo", "pdf", "pgf", "ps", "svg", "template"]:
                self.show_plot = False
            else:
                self.show_plot = True
        else:
            self.show_plot = show_plot
        if self.step is None or self.step == 0:
            self.step = 1
        if self.savegif == "":
            self.savegif == None
        if self.slice is None:
            self.run()
        if self.slice is not None:
            self.slicer()

    def run(self):
        if self.dim not in self.dimlist:
            raise NameError("check inputs for dim, should be string.")

        if self.dim in [None, "X", "x"]:
            self.dimint = 2
            self.dimlist = ["->Y", "->Z", "X"]
            self.run_plot()
        if self.dim in ["Y", "y"]:
            self.dimint = 1
            self.dimlist = ["->X", "->Z", "Y"]
            self.run_plot()
        if self.dim in ["Z", "z"]:
            self.dimint = 0
            self.dimlist = ["->X", "->Y", "Z"]
            self.run_plot()

    def update_frame(self, it, fig, min_val, max_val):
        i = range(0, self.cube.shape[self.dimint])[:: self.step][it]
        fig.clf()
        axis = fig.add_subplot(1, 1, 1)
        if self.dimint == 2:
            mappable = axis.imshow(
                np.squeeze(self.cube[:, :, i]),
                cmap=self.colormap,
                origin="lower",
                vmin=self.min_val,
                vmax=self.max_val,
            )
        if self.dimint == 1:
            mappable = axis.imshow(
                np.squeeze(self.cube[:, i]),
                cmap=self.colormap,
                origin="lower",
                vmin=self.min_val,
                vmax=self.max_val,
            )
        if self.dimint == 0:
            mappable = axis.imshow(
                np.squeeze(self.cube[i]),
                cmap=self.colormap,
                origin="lower",
                vmin=self.min_val,
                vmax=self.max_val,
            )
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.set_xlabel(self.dimlist[0])
        axis.set_ylabel(self.dimlist[1])
        axis.set_title(self.dimlist[2] + ":" + str(i))
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(mappable, cax=cax)
        # plt.pause(0.01)

    def run_plot(self):

        dim = self.cube.shape

        fig = plt.figure()
        ani = animation.FuncAnimation(
            fig,
            self.update_frame,
            fargs=(fig, self.min_val, self.max_val),
            interval=100,
            repeat_delay=1000,
            frames=len(range(0, dim[self.dimint])[:: self.step]),
        )
        if self.savegif is not None:
            ani.save(self.savegif, writer="pillow")
            self._show()
        else:
            self._show()

    def slicer(self):

        if self.dim in [None, "X", "x"]:
            plt.xlabel("Y")
            plt.ylabel("Z")
            plt.imshow(
                np.squeeze(self.cube[:, :, self.slice]),
                cmap=self.colormap,
                origin="lower",
                vmin=self.min_val,
                vmax=self.max_val,
            )
        if self.dim in ["Y", "y"]:
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.imshow(
                np.squeeze(self.cube[:, self.slice]),
                cmap=self.colormap,
                origin="lower",
                vmin=self.min_val,
                vmax=self.max_val,
            )
        if self.dim in ["Z", "z"]:
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.imshow(
                np.squeeze(self.cube[self.slice]),
                cmap=self.colormap,
                origin="lower",
                vmin=self.min_val,
                vmax=self.max_val,
            )
        self._show()

    def _show(self):
        if self.show_plot:
            plt.show()

# """Test for addPossionGaussianNoisy and plotImg3D"""
# Proj = np.zeros([300, 300, 60], dtype=np.float32)
# Proj[100:200, 100:200, :] = 1
# NoisyProj = addPossionGaussianNoisy(Proj, Poisson=1e4, Gaussian=np.array([0, 10]))
# # import pylab
# # pylab.imshow(NoisyProj[:, :, 30] - Proj[:, :, 30], cmap='gray')
# # import tigre
# plotImg3D(NoisyProj - Proj, dim="x")
