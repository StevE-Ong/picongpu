#!/usr/bin/env python
#
# Copyright 2013-2023 Richard Pausch
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

import matplotlib
import os
import re
import warnings
import numpy as np
from matplotlib.colors import LogNorm
import argparse
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import pylab

__doc__ = """
This program interprets the spectral data generated by the Radiation plugin
or CLARA 2.0. The spectra can be shown in a window or stored as a pdf file.
Plotted are:
   x-axis: frequencies,
   y-axis: directions,
   color:  intensity

Developer: Richard Pausch
"""


# ------ check python environment --------
"""
Check if the running version might cause problems. A typical problem
is matplotlib. Several arguments for plots differ between versions.
Therefore a warning will be given if a different matplotlib version
is used.
"""


tested_matplotlib_version = ["1.2.0", "1.3.1", "2.0.0rc1"]
if matplotlib.__version__ not in tested_matplotlib_version:
    warning_msg = """The matplotlib version differs from that used during
                     development. This might cause the output to appear
                     different.
                     Version used: {}, tested versions {}""".format(matplotlib.__version__, tested_matplotlib_version)
    warnings.warn(warning_msg)

# ------ argsparse --------
"""
The following part of the code describes the shell user interface.
Several command line options are defined, f.e. the path to the input,
windows or storing options, labels, etc..
"""
parser = argparse.ArgumentParser(description=__doc__, epilog="For further questions please contact Richard Pausch.")

# Path to input files (several are possible) - necessary argument
parser.add_argument(
    "path2Data",
    metavar="Path to input file",
    type=argparse.FileType("r"),
    nargs="+",
    help="location of input file (giving several \
                    files is possible)",
)

# Default windows, but pdf version can be switched on.
parser.add_argument(
    "--pdf",
    dest="outputPDF",
    action="store_true",
    help="store output as pdf instead of using window",
)

# gives destination of  output (for several input files,
# a number will be added)
parser.add_argument(
    "--output",
    "-o",
    dest="outputName",
    metavar="path",
    nargs="?",
    default="SpectraOutput",
    help="""Path for output  if --pdf was chosen. \
                    [default=SpectraOutput]
                    For many input files: file name will be added \
                    between output name and .dat""",
)

# extend of the input data
parser.add_argument(
    "--dataExtend",
    "-d",
    nargs=4,
    metavar="float",
    type=float,
    default=(0.0, 1.0, 0.0, 90.0),
    help="""extend of the input data: omega_min omega_max,
                            theta_min, theta_max [default=0.0 1.0 0.0 90.0]""",
)

# extend of the input data
parser.add_argument(
    "--outputExtend",
    "-z",
    nargs=4,
    metavar="float",
    type=float,
    default=[],
    help="""extend of the output data: omega_min omega_max,
                            theta_min, theta_max [default= as dataExtend]""",
)

# This flag sets output of dI^2/domega dOmega to log scale.
parser.add_argument(
    "--logInt",
    dest="logIntensity",
    action="store_true",
    help="""sets output of dI^2/domega dOmega to log scale""",
)

# This flag sets omega-axis to log scale.
parser.add_argument(
    "--logOmega",
    dest="logOmega",
    action="store_true",
    help="""sets omega-axis to log scale""",
)

# set label x-axis
parser.add_argument(
    "--labelOmega",
    "-x",
    metavar="string",
    default="$\\omega/\\omega_0$",
    help="""Label text for omega axis (x-axis) in LaTeX style.
                            [default: $\\omega/\\omega_0$]""",
)

# set label y-axis
parser.add_argument(
    "--labelTheta",
    "-y",
    metavar="string",
    default="$\\theta/^\\circ$",
    help="""Label text for theta axis (y-axis) in LaTeX style.
                            [default: $\\\\theta/^\\circ$]""",
)

# set label c-axis
parser.add_argument(
    "--labelColorbar",
    metavar="string",
    default="$\\frac{\\mathrm{d} ^2I}{\\mathrm{d} \
                             \\omega \\mathrm{d} \\Omega}/$Js",
    help="""Label text for colorbar (c-axis) in LaTeX style.
                            [default: $\\\\frac{\\\\mathrm{d} ^2I}
                            {\\\\mathrm{d} \\\\omega \\\\mathrm{d}
                            \\\\Omega}/$Js$]""",
)

# set maximum value for colorbar
parser.add_argument(
    "--vMax",
    dest="colorbarMax",
    nargs="?",
    metavar="float",
    type=float,
    default=[],
    help="""maximum value shown in colorbar (applied
                            after smoothing) [default: actual maximum]""",
)

# set maximum value for data
parser.add_argument(
    "--dataMax",
    dest="dataMax",
    nargs=1,
    metavar="float",
    type=float,
    default=[-1],
    help="""maximum value of data used (applied
                            before smoothing) [default: not used]""",
)

# set on and configure smoothing of data
parser.add_argument(
    "--smooth",
    dest="smooth",
    nargs=2,
    metavar="float",
    type=float,
    default=[-1.0, -1.0],
    help="""switch on smoothing of data with sigma_omega/x and
                            sigma_theta/y in bins [default: no smoothing]""",
)

# set on and configure smoothing of data
parser.add_argument(
    "--split",
    dest="split",
    nargs=2,
    metavar="float",
    type=float,
    default=[-1.0, -1.0],
    help="""select between different ranges of observation
                            angles, param1: first index theta, param2:
                            last index theta +1
                            [default: only one range is assumed]""",
)

# Default rainbow colorbar, but a black&white colorbar can be needed
# for publications.
parser.add_argument(
    "--bw",
    dest="colorbarBlackAndWhite",
    action="store_true",
    help="""use black and white colorbar instead of using
                            rainbow colors""",
)


# Default gouraud color interpolation, but a flat/nearest interpolation can
# be need for tests.
parser.add_argument(
    "--nearest",
    dest="interpolSet",
    action="store_true",
    help="""use flat/nearest interpolation instead of
                            gouraud""",
)


# get arguments and store them in args
args = parser.parse_args()


# ------ work with program arguments ------
"""
fill arguments not set yet
rename arguments  for easier use
check arguments consistency
"""

# in the case of no outputExtend, set outExtend equal to dataExtend
if not args.outputExtend:
    args.outputExtend = args.dataExtend


# set extent data:
omega_min_data = args.dataExtend[0]
omega_max_data = args.dataExtend[1]
theta_min_data = args.dataExtend[2]
theta_max_data = args.dataExtend[3]

# set extent picture:
omega_min_draw = args.outputExtend[0]
omega_max_draw = args.outputExtend[1]
theta_min_draw = args.outputExtend[2]
theta_max_draw = args.outputExtend[3]

# set flags:
Nfiles = len(args.path2Data)
manyFiles = False
if Nfiles > 1:
    manyFiles = True

# set scaling function of color plot
if args.logIntensity:
    colorNorm = LogNorm()
else:
    colorNorm = None

# check if extents are consistent
if args.logOmega:
    if min(omega_min_data, omega_min_draw, omega_max_data, omega_max_draw) <= 0:
        raise Exception("omega <= 0 is not allowed")
else:
    if min(omega_min_data, omega_min_draw, omega_max_data, omega_max_draw) < 0:
        raise Exception("omega < 0 is not allowed")


# set interpolation
if args.interpolSet:
    my_interpolation = "flat"
else:
    my_interpolation = "gouraud"


# ------ setup graphic output -------

if args.outputPDF:
    matplotlib.use("Agg")

# X-window/pdf size
window_width = 14.4
window_height = 9

# font sizes
tickfontsize = 28
labelfontsize = 39
labelfontsize_colorbar = 45
exponentsize = 18

# distance between ticks and label
pylab.rcParams["xtick.major.pad"] = "10"
pylab.rcParams["ytick.major.pad"] = "10"


# create my color map
cdict_color = {
    "red": (
        (0.0, 1, 1),
        (0.03, 0, 0),
        (0.35, 0, 0),
        (0.66, 1, 1),
        (0.89, 1, 1),
        (1, 0.5, 0.5),
    ),
    "green": (
        (0.0, 1, 1),
        (0.03, 0, 0),
        (0.125, 0, 0),
        (0.375, 1, 1),
        (0.64, 1, 1),
        (0.91, 0, 0),
        (1, 0, 0),
    ),
    "blue": (
        (0.0, 1, 1),
        (0.03, 0.8, 0.8),
        (0.11, 1, 1),
        (0.34, 1, 1),
        (0.65, 0, 0),
        (1, 0, 0),
    ),
}

# create my black and white map
cdict_bw = {
    "red": ((0.0, 1, 1), (0.03, 0.7, 0.7), (1, 0.0, 0.0)),
    "green": ((0.0, 1, 1), (0.03, 0.7, 0.7), (1, 0.0, 0.0)),
    "blue": ((0.0, 1, 1), (0.03, 0.7, 0.7), (1, 0.0, 0.0)),
}

if args.colorbarBlackAndWhite:
    cdict = cdict_bw
else:
    cdict = cdict_color

my_cmap = colors.LinearSegmentedColormap("my_colormap", cdict, 256)


# ------- run through all files and create spectral plots -------

for myfile in args.path2Data:
    # load ascii data files
    data = np.loadtxt(myfile)

    # select only theta range of interest
    if args.split[0] != -1 and args.split[1] != -1:
        data = data[args.split[0] : args.split[1]]

    # find indices for zoomed plot
    if args.logOmega:
        omega_max_index = int(
            (np.log(omega_max_draw) - np.log(omega_min_data))
            / (np.log(omega_max_data) - np.log(omega_min_data))
            * data.shape[1]
        )
        omega_min_index = int(
            (np.log(omega_min_draw) - np.log(omega_min_data))
            / (np.log(omega_max_data) - np.log(omega_min_data))
            * data.shape[1]
        )
        theta_max_index = int((theta_max_draw - theta_min_data) / (theta_max_data - theta_min_data) * data.shape[0])
        theta_min_index = int((theta_min_draw - theta_min_data) / (theta_max_data - theta_min_data) * data.shape[0])
    else:
        omega_max_index = int((omega_max_draw - omega_min_data) / (omega_max_data - omega_min_data) * data.shape[1])
        omega_min_index = int((omega_min_draw - omega_min_data) / (omega_max_data - omega_min_data) * data.shape[1])
        theta_max_index = int((theta_max_draw - theta_min_data) / (theta_max_data - theta_min_data) * data.shape[0])
        theta_min_index = int((theta_min_draw - theta_min_data) / (theta_max_data - theta_min_data) * data.shape[0])

    # reduction to zoomed values
    dataZoomed = data[theta_min_index:theta_max_index, omega_min_index:omega_max_index]

    # reduction to maximum value
    if args.dataMax[0] != -1:
        dataZoomed = np.minimum(args.dataMax[0], dataZoomed)

    # smoothing data:
    if args.smooth[0] != -1:
        import smooth

        sigma_omega = args.smooth[0]
        sigma_theta = args.smooth[1]
        len_omega_smooth = int(5.0 * sigma_omega)
        len_theta_smooth = int(5.0 * sigma_theta)
        dataZoomed = smooth.smooth2D(dataZoomed, sigma_omega, len_omega_smooth, sigma_theta, len_theta_smooth)

    # figure environment (is needed to be deleted for next file)
    FG = plt.figure("Plot Radiation", figsize=(window_width, window_height))

    # subplot environment
    SP = plt.subplot(111, autoscale_on=True)
    # labels
    plt.xlabel(args.labelOmega, fontsize=labelfontsize)
    plt.xticks(size=tickfontsize)
    plt.ylabel(args.labelTheta, fontsize=labelfontsize)
    plt.yticks(size=tickfontsize)
    SP.set_yticks(np.linspace(theta_min_draw, theta_max_draw, 5))

    # find max for colorbar
    if not args.colorbarMax:
        maxData = dataZoomed.max()
        colorbarMax = maxData
    else:
        colorbarMax = args.colorbarMax

    # set omega-(x)-scale to log if needed
    shape = np.shape(dataZoomed)
    if args.logOmega:
        SP.set_xscale("log")
        x = np.logspace(np.log10(omega_min_draw), np.log10(omega_max_draw), shape[1])
    else:
        x = np.linspace(omega_min_draw, omega_max_draw, shape[1])

    y = np.linspace(theta_min_draw, theta_max_draw, shape[0])

    # plot data: (this is the only command that plots data)
    cax = SP.pcolormesh(
        x,
        y,
        dataZoomed,
        shading=my_interpolation,
        cmap=my_cmap,
        vmax=colorbarMax,
        norm=colorNorm,
    )

    # set x and y range (required after pcolormesh(?))
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])

    # set format style of colorbar label
    if args.logIntensity:
        colorbarFormat = None
    else:
        colorbarFormat = "%2.2e"

    # create  colorbar
    CB = plt.colorbar(cax, format=colorbarFormat, pad=0.04)
    CB.set_label(args.labelColorbar, fontsize=labelfontsize_colorbar)

    # re-size label of colorbar
    for t in CB.ax.get_yticklabels():
        t.set_fontsize(tickfontsize)

    CB.ax.yaxis.get_offset_text().set_fontsize(exponentsize)

    # set appearance of plot
    plt.tight_layout()
    # get name of file
    filename = os.path.basename(myfile.name)

    # store or show plot
    if args.outputPDF:
        # generate template for multiple file output
        filename_template = filename[0 : filename.rfind(".")]
        filename_template = re.sub(r"\d+", "{:09d}", filename_template)

        # extract index
        index_filename = int(re.findall(r"\d+", filename)[0])

        # one or many file?
        if manyFiles:
            filename_temp = filename_template.format(index_filename)
            filename_temp = "{}_{}.pdf".format(args.outputName, filename_temp)
            plt.savefig(filename_temp, format="pdf")
        else:
            plt.savefig(args.outputName + ".pdf", format="pdf")
    else:
        plt.show()

    # clear everything to restart with next  file
    FG.clear()
    myfile.close()

    print(filename + "\t Done")
