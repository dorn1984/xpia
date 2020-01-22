import numpy as np
import os, glob
from netCDF4 import Dataset

import timeit
from pandas import to_datetime, date_range
import wrf, xarray, sys
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### define paths

dataPath = '/glade/scratch/doubrawa/final_data/les/'
outPath  = '/glade/scratch/doubrawa/post_processing_new/'

### define some parameters

n_processors       = 1800
domainId           = 4
prefix             = "LES_25m" if domainId==4 else "LES_100m"

### horizontal size of staggered and non-staggered grids

n_east_west        = 1200
n_east_west_stag   = n_east_west + 1
n_south_north      = 1200
n_south_north_stag = n_south_north + 1

### variables of interest, and how they will be called in the wrfout file

xarray_mapping = {"w":"wa","u":"U","v":"V","theta":"theta"}
units          = {"w":"m s-1","u":"m s-1","v":"m s-1","theta":"K"}
names          = {"w":"vertical velocity","u":"zonal velocity","v":"meridional velocity","theta":"potential temperature"}

## total desired variables

variables_static = ["XLAT","XLONG","ter"]
variables_3d     = ["U","V","wa","theta","z"]
variables        = variables_3d

## how many vertical levels to keep?
kmax = 41

## what day to focus on
day  = 20

# loop through all hours!
for hour in range(14,24,1):

    desired_times = pd.date_range(start='2015-03-{1:d} {0:d}:00'.format(hour,day),end='2015-03-{1:d} {0:d}:59'.format(hour,day), freq='10min')

    # find out how many history files there are per processor in this folder
    file_paths = sorted(glob.glob(os.path.join(dataPath,
                 "03{0:d}15".format(day,hour),
                 "03{0:d}15_{1:d}UTC".format(day,hour),
                 'wrfout_d0{0}*_0000'.format(domainId))))

    # also consider files of the previous hour, sometimes something like 15:00:00 is in the file of the 14:30:00 instead
    file_paths_to_append = sorted(glob.glob(os.path.join(dataPath,
                 "03{0:d}15".format(day,hour-1),
                 "03{0:d}15_{1:d}UTC".format(day,hour-1),
                 'wrfout_d0{0}*_0000'.format(domainId))))

    for f in file_paths_to_append:
        file_paths.append(f)

    # for each desired timestamp, find out which ncfile to open
    map_desired_time_to_ncfile = {}
    for file_path in file_paths:
        wrfnc = xarray.open_dataset(file_path)
        wrf_datetimes = np.asarray([ to_datetime(ii) for ii in wrfnc.XTIME.data ])
        for desired_time in desired_times:
            deltas = np.abs([ (desired_time - wd).total_seconds() for wd in wrf_datetimes])
            idx = np.argmin(deltas)
            minute = desired_time.minute
            dt_accept = 15 if ( (hour==14)&(minute==0)) else 5
            if (np.any(deltas<dt_accept)):
                map_desired_time_to_ncfile[desired_time] = file_path

    # only once every time we run this code, we will need to read in the static files;
    # allocate space here
    data_static = {}
    for var in variables_static:
        data_static[var] = np.zeros((n_south_north,n_east_west))

    # keep track on whether it's the first time or not via this logic variable
    first = True

    # start time loop (every 10 minutes in this hour)
    for desired_time in sorted(map_desired_time_to_ncfile.keys()):

        print('----------------------')
        file_prefix = map_desired_time_to_ncfile[desired_time][0:-4]

        # for this time, allocate space
        data         = {}
        for var in ["U"]:
            data[var] = np.zeros((kmax,n_south_north,n_east_west_stag))
        for var in ["V"]:
            data[var] = np.zeros((kmax,n_south_north_stag,n_east_west))
        for var in ["wa","theta","z"]:
            data[var] = np.zeros((kmax,n_south_north,n_east_west))

        for processor in range(n_processors):

            file_name = glob.glob(file_prefix+"{0:04d}".format(processor))[0]

            # print out which file is being read
            sys.stdout.write('\r'+file_name)

            # open the netcdf file with xarray
            wrfnc = xarray.open_dataset(file_name)

            # open it in a different way also to use the wrf package
            wrfnc_for_wrf = Dataset(file_name,'r')

            # find out what index corresponds to the desired time
            if processor==0:
                wrf_datetimes = np.asarray([ to_datetime(ii) for ii in wrfnc.XTIME.data ])
                dt_between_desired_and_actual = np.min([ ii.seconds for ii in (wrf_datetimes - desired_time) ])
                dt_idx = np.argmin([ ii.seconds for ii in (wrf_datetimes - desired_time) ])

            # for this time and this processor, get all the variables:
            for var in variables:

                try:
                    data_tmp = wrf.getvar(wrfnc_for_wrf, var, timeidx=dt_idx).data
                except:
                    data_tmp = wrfnc[var].isel(Time=dt_idx).data

                # the minus one is for python indexing...
                we_0 = getattr(wrfnc,'WEST-EAST_PATCH_START_UNSTAG') - 1
                we_1 = getattr(wrfnc,'WEST-EAST_PATCH_END_UNSTAG')

                # the minus one is for python indexing...
                sn_0 = getattr(wrfnc,'SOUTH-NORTH_PATCH_START_UNSTAG') - 1
                sn_1 = getattr(wrfnc,'SOUTH-NORTH_PATCH_END_UNSTAG')

                if data_tmp.ndim==3:
                    # if it's U then overwrite the we dimensions b/c it's staggered along we
                    if var=='U':
                        we_0 = getattr(wrfnc,'WEST-EAST_PATCH_START_STAG') - 1
                        we_1 = getattr(wrfnc,'WEST-EAST_PATCH_END_STAG')
                    # if it's V then overwrite the sn dimensions b/c it's staggered along sn
                    if var=='V':
                        sn_0 = getattr(wrfnc,'SOUTH-NORTH_PATCH_START_STAG') - 1
                        sn_1 = getattr(wrfnc,'SOUTH-NORTH_PATCH_END_STAG')
                    # the only other options are theta, w, and z which are not staggered in we or sn so nothing needs to be done
                    data[var][:, sn_0:sn_1, we_0:we_1] = data_tmp[0:kmax,:,:].copy()
                else:
                    # or if the variable is 2d, then don't need the first dimension
                    data[var][sn_0:sn_1, we_0:we_1] = data_tmp.copy()

            # only once every time we run this code, we will need to read in the static files
            if first:
                for var in variables_static:
                    try:
                        data_tmp = wrf.getvar(wrfnc_for_wrf, var, timeidx=dt_idx).data
                    except:
                        data_tmp = wrfnc[var].isel(Time=dt_idx).data

                    we_0 = getattr(wrfnc,'WEST-EAST_PATCH_START_UNSTAG') - 1
                    we_1 = getattr(wrfnc,'WEST-EAST_PATCH_END_UNSTAG')

                    sn_0 = getattr(wrfnc,'SOUTH-NORTH_PATCH_START_UNSTAG') - 1
                    sn_1 = getattr(wrfnc,'SOUTH-NORTH_PATCH_END_UNSTAG')

                    data_static[var][sn_0:sn_1, we_0:we_1] = data_tmp.copy()

        # remove terrain from z
        data["z"] = data["z"] - data_static["ter"]

        # unstagger u and v
        data["U"] = 0.5*(data["U"][:,:,0:n_east_west_stag-1] + data["U"][:,:,1:n_east_west_stag+1])
        data["V"] = 0.5*(data["V"][:,0:n_south_north_stag-1,:] + data["V"][:,1:n_south_north_stag+1,:])

        # Prepare xarrays of two-dimensional variables: XLAT, XLONG, TER
        #
        if first:
            n_sn, n_we = data_static['XLAT'].shape
            xlat = xarray.DataArray(data_static['XLAT'],
                             coords={"south_north":range(n_sn),"west_east":range(n_we)},
                             dims=("south_north","west_east"),
                             name="2-d latitude",
                             attrs={"unit":"deg","stagger":""})

            xlong = xarray.DataArray(data_static['XLONG'],
                             coords={"south_north":range(n_sn),"west_east":range(n_we)},
                             dims=("south_north","west_east"),
                             name="2-d longitude",
                             attrs={"unit":"deg","stagger":""})
            terrain = xarray.DataArray(data_static['ter'],
                             coords={"south_north":range(n_sn),"west_east":range(n_we)},
                             dims=("south_north","west_east"),
                             name="terrain height",
                             attrs={"unit":"m","stagger":""})
            first = False

        # Prepare xarrays of two-dimensional variables: U, V, z, wa, theta
        #
        xarray_zref = xarray.DataArray(data["z"], \
                                       coords={"bottom_top":range(kmax),"south_north":range(n_sn),"west_east":range(n_we)},  \
                                       dims=("bottom_top","south_north","west_east"), \
                                       name="height above ground", \
                                       attrs={"unit":"m","stagger":""})

        xarray_zref["lat"] = xlat
        xarray_zref["lon"] = xlong

        xarray_u = xarray.DataArray(data["U"], \
                                       coords={"bottom_top":range(kmax),"south_north":range(n_sn),"west_east":range(n_we)},  \
                                       dims=("bottom_top","south_north","west_east"), \
                                       name="zonal wind", \
                                       attrs={"unit":"m/s","stagger":""})

        xarray_u["lat"] = xlat
        xarray_u["lon"] = xlong

        xarray_v = xarray.DataArray(data["V"], \
                                       coords={"bottom_top":range(kmax),"south_north":range(n_sn),"west_east":range(n_we)},  \
                                       dims=("bottom_top","south_north","west_east"), \
                                       name="meridional wind", \
                                       attrs={"unit":"m/s","stagger":""})

        xarray_v["lat"] = xlat
        xarray_v["lon"] = xlong

        xarray_w = xarray.DataArray(data["wa"], \
                                       coords={"bottom_top":range(kmax),"south_north":range(n_sn),"west_east":range(n_we)},  \
                                       dims=("bottom_top","south_north","west_east"), \
                                       name="vertical wind", \
                                       attrs={"unit":"m/s","stagger":""})

        xarray_w["lat"] = xlat
        xarray_w["lon"] = xlong

        xarray_theta = xarray.DataArray(data["theta"], \
                                       coords={"bottom_top":range(kmax),"south_north":range(n_sn),"west_east":range(n_we)},  \
                                       dims=("bottom_top","south_north","west_east"), \
                                       name="potential temperature", \
                                       attrs={"unit":"K","stagger":""})

        xarray_theta["lat"] = xlat
        xarray_theta["lon"] = xlong


        ## combine all these xarrays into a Dataset
        xarray_dict = {"xlat":xlat, "xlong":xlong, "terrain":terrain, "z":xarray_zref, "u":xarray_u, "v":xarray_v, "w":xarray_w, "theta":xarray_theta}
        coords_dict = {"bottom_top":range(kmax),"south_north":range(n_sn),"west_east":range(n_we)}
        attrs_dict  = {"valid":"{0:%Y-%m-%d_%H:%M}".format(wrf_datetimes[dt_idx])}
        data_set    = xarray.Dataset(data_vars=xarray_dict,coords=coords_dict,attrs=attrs_dict)

        # save dataset into a netcdf file ==> for filtering later on!
        fName   = "WRF_{0}_3D_{1:%Y-%m-%d_%H:%M}.nc".format(prefix,wrf_datetimes[dt_idx])
        fPath   = os.path.join(outPath,fName)
        print("Saving : {0}".format(fName))
        data_set.to_netcdf(fPath)



