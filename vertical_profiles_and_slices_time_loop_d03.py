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

dataPath = '/glade/scratch/domingom/Cheyenne/XPIA_mesoLES/SIMULS/'
outPath  = '/glade/scratch/doubrawa/postProcessing/'

### define some parameters

n_processors       = 1800
domainId           = 3
prefix             = "LES_25m" if domainId==4 else "LES_100m"

nt_per_file        = 180 if domainId==4 else 30
nhours             = 10
nfiles             = nhours * 2
nt_total           = nt_per_file * nfiles

n_east_west        = 1200
n_east_west_stag   = n_east_west + 1
n_south_north      = 1200
n_south_north_stag = n_south_north + 1


xarray_mapping = {"w":"wa","u":"U","v":"V","theta":"theta"}
units = {"w":"m s-1","u":"m s-1","v":"m s-1","theta":"K"}
names = {"w":"vertical velocity","u":"zonal velocity","v":"meridional velocity","theta":"potential temperature"}   

### reference wrf to clip domains
ref_wrfnc = Dataset("/glade/scratch/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_12_mesoLES/HOUR_14_1/wrfout_d04_2015-03-13_14:00:10_0000")

## desired variables 

variables_static = ["XLAT","XLONG","ter"]
variables_2d     = ["RMOL","HFX","UST","LH","QFX"]
variables_3d     = ["U","V","wa","theta","z"]
variables        = variables_2d + variables_3d
    
## open all data

year           = 2018
month          = 3
day            = 21
first          = True

for day in [day]:
    for hour in np.arange(14,25,1):
        if hour==24:
            minutes = [0]
        else:
            minutes = [0,30]
        for minute in minutes:

            #
            # Based on desired time, find out which file to actually open!
            #
            if (hour==14)&(minute==0):
                # at the initial time, we will invariably be off by dt because the
                # 14:00:00 file actually starts logging at 14:00:00 + dt and the
                # 14:00:00 data would only have been found in a 13:30:00 file which
                # I do not have
                file_hour   = hour
                file_minute = minute
            else:
                file_hour   = hour-1 if minute==0 else hour
                file_minute = 0 if minute==30 else 30
            half_hour_idx  = 1 if file_minute<30 else 2   

            file_day = day
            if hour==24:
                day += 1
                hour = 0
                
            datetime       = to_datetime("{0}-{1}-{2} {3}:{4}".format(year,month,day,hour,minute),format="%Y-%m-%d %H:%M")
            file_datetime  = to_datetime("{0}-{1}-{2} {3}:{4}".format(year,month,file_day,file_hour,file_minute),format="%Y-%m-%d %H:%M")            

            print ("Seeking {0} -- will open {1}".format(datetime, file_datetime))    
            
            #
            # Allocate space
            #
            data         = {}
            for var in ["U"]:
                data[var] = np.zeros((72,n_south_north,n_east_west_stag))
            for var in ["V"]:
                data[var] = np.zeros((72,n_south_north_stag,n_east_west))    
            for var in variables_2d:
                data[var] = np.zeros((1200,1200))
            for var in ["wa","theta","z"]:
                data[var] = np.zeros((72,n_south_north,n_east_west))        

            #
            # Static variables
            #
            if first:
                data_static = {}
                for var in variables_static:
                    data_static[var] = np.zeros((1200,1200))                
                
            #
            # Start patching
            #
            for processor in range(n_processors):
                wrfoutPath     = (glob.glob(os.path.join(dataPath,
                                               'WRF_mesoLES_4dom_RAP_2015_03_{0}_mesoLES'.format(file_day-1),
                                               'HOUR_{0}_{1}'.format(file_hour,half_hour_idx),
                                               'wrfout_d0{0}_*_{1:04d}'.format(domainId,processor))))[0]
                
                #
                # Open file
                #
                wrfnc          = xarray.open_dataset(wrfoutPath)
                wrfnc_for_wrf  = Dataset(wrfoutPath,'r')
                wrf_datetimes  = np.asarray([ to_datetime(ii) for ii in wrfnc.XTIME.data ])
                
                if processor==0:
                    dt_between_desired_and_actual = np.min([ ii.seconds for ii in (wrf_datetimes - datetime) ])
                    dt_idx         = np.argmin([ ii.seconds for ii in (wrf_datetimes - datetime) ])
                    print (" ")
                    print ("Time offset between desired and available datetime = {0} seconds".format(dt_between_desired_and_actual))
                    print (" ")
                
                #
                # get all variables (3-d in space) 
                #
                for var in variables:

                    try:
                        data_tmp = wrf.getvar(wrfnc_for_wrf, var, timeidx=dt_idx).data
                    except:
                        data_tmp = wrfnc[var].isel(Time=dt_idx).data       

                    we_0 = getattr(wrfnc,'WEST-EAST_PATCH_START_UNSTAG') - 1        
                    we_1 = getattr(wrfnc,'WEST-EAST_PATCH_END_UNSTAG')                

                    sn_0 = getattr(wrfnc,'SOUTH-NORTH_PATCH_START_UNSTAG') - 1       
                    sn_1 = getattr(wrfnc,'SOUTH-NORTH_PATCH_END_UNSTAG')               



                    if data_tmp.ndim==3:
                        if var=='U':
                            we_0 = getattr(wrfnc,'WEST-EAST_PATCH_START_STAG') - 1        
                            we_1 = getattr(wrfnc,'WEST-EAST_PATCH_END_STAG')                                
                        if var=='V':
                            sn_0 = getattr(wrfnc,'SOUTH-NORTH_PATCH_START_STAG') - 1       
                            sn_1 = getattr(wrfnc,'SOUTH-NORTH_PATCH_END_STAG')                                                                       
                        data[var][:, sn_0:sn_1, we_0:we_1] = data_tmp.copy()
                    else:
                        data[var][sn_0:sn_1, we_0:we_1] = data_tmp.copy()            

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
                                 
            data["z"] = data["z"] - data_static["ter"]

            #
            # unstagger u and v
            #
            data["U"] = 0.5*(data["U"][:,:,0:n_east_west_stag-1] + data["U"][:,:,1:n_east_west_stag+1])
            data["V"] = 0.5*(data["V"][:,0:n_south_north_stag-1,:] + data["V"][:,1:n_south_north_stag+1,:])

            #
            # clip to the area of d04
            #
            if first:
                x,y = wrf.ll_to_xy(ref_wrfnc, data_static["XLAT"], data_static["XLONG"])    
                x   = np.reshape(x.data,data["wa"].shape[1:])
                y   = np.reshape(y.data,data["wa"].shape[1:])    
                idx_j, idx_i = np.where ( (x>=0) & (y>=0) & (x<=1200) & (y<=1200) )
                j0, j1 = np.min(idx_j), np.max(idx_j)
                i0, i1 = np.min(idx_i), np.max(idx_i)
                
            for var in data:
                ndim = (len(data[var].shape))
                if ndim==3:
                    data[var] = data[var][:,j0:j1+1,i0:i1+1].copy()
                if ndim==2:
                    data[var] = data[var][j0:j1+1,i0:i1+1].copy()        

            if first:
                for var in data_static: 
                    data_static[var] = data_static[var][j0:j1+1,i0:i1+1].copy()                     
                    
            #               
            # get profile of planar averages
            #
            data_mean = {}
            for var in ["U","V","wa","theta","z"]:
                data_mean[var] = np.mean(data[var],axis=(1,2))

            #              
            # get profile of planar perturbations
            #
            data_prime  = {}
            for var in data_mean.keys():
                data_prime[var] = data[var] - data_mean[var][:,None,None]   

            #
            # compute fluxes
            #
            fluxes = ["U_U","V_V","wa_wa","U_wa","V_wa","U_V","wa_theta"]
            data_fluxes = {}
            for flux in fluxes:
                var1 = flux.split("_")[0]
                var2 = flux.split("_")[1]
                data_fluxes[flux] = np.mean(data_prime[var1]*data_prime[var2],axis=(1,2))
                           
            #
            # organize them into a dataframe and save
            #
            df  = pd.DataFrame(data_mean).set_index("z")
            df["z_std_xy"] = np.std(data["z"],axis=(1,2))
            df2 = pd.DataFrame(data_fluxes).set_index(df.index)
            df  = pd.concat([df,df2],axis=1)
            column_mapping = {"U":"u",
             "V":"v",
             "wa":"w",
             "theta":"theta",
             "z_std_xy":"z_std_xy",
             "U_U":"u_u",
             "V_V":"v_v",
             "wa_wa":"w_w",
             "U_wa":"u_w",
             "V_wa":"v_w",
             "U_V":"u_v",
             "wa_theta":"w_theta",
             "wa_theta0":"w_theta0"}
            df.columns = [ column_mapping[col_old] for col_old in df.columns ]
            fName = os.path.join(outPath,"WRF_{0}_CLIPPED_SPATIAL_AVERAGED_PROFILES_{1:%Y-%m-%d_%H:%M:%S}.csv".format(prefix,datetime))
            print(fName)
            df.to_csv(fName)
                           
            #
            # Save the 2d stuff too 
            #
            means_2d = {}
            for var in ["RMOL","HFX"]:
                means_2d[var] = np.mean(data[var])
            a = pd.Series(means_2d)
            fName = os.path.join(outPath,"WRF_{0}_CLIPPED_SPATIAL_AVERAGED_2D_{1:%Y-%m-%d_%H:%M:%S}.csv".format(prefix,datetime))   
            a.to_csv(fName)

            #
            # Prepare planes of XLAT, XLONG
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
            first = False


            #
            # Prepare z plane for vertical interpolation of other variables
            #            
            xarray_zref = xarray.DataArray(data["z"], \
                                           coords={"bottom_top":range(72),"south_north":range(n_sn),"west_east":range(n_we)},  \
                                           dims=("bottom_top","south_north","west_east"), \
                                           name="height above ground", \
                                           attrs={"unit":"m","stagger":""})

            xarray_zref["lat"] = xlat
            xarray_zref["lon"] = xlong           
            
            #
            # Interpolate other variables to desired heights
            #
            heights = [100.0,500.0]            
            for height in heights:
                xarray_dict = {}         
                for xarray_varname in ["w","u","v","theta"]:
                    print(xarray_varname)
                    xarray_3d = xarray.DataArray(data[xarray_mapping[xarray_varname]][None,:,:,:],     \
                                                   coords={  "time":[wrf_datetimes[dt_idx]],"bottom_top":range(72),"south_north":range(n_sn),"west_east":range(n_we)}, \
                                                   dims=("time","bottom_top","south_north","west_east"), 
                                                   name=names[xarray_varname],
                                                   attrs={"unit":units[xarray_varname],"stagger":"","height [m]":height})

                    xarray_3d["lat"] = xlat
                    xarray_3d["lon"] = xlong

                    var_3d_now = data[xarray_mapping[xarray_varname]]
                    xarray_dict[xarray_varname] = wrf.interplevel(xarray_3d, xarray_zref, height, meta=True)
                    xarray_dict[xarray_varname]["z"] = height
                dataset = xarray.Dataset(xarray_dict)
                fName   = "WRF_{0}_{1}_m_AGL_{2:%Y-%m-%d_%H:%M}.nc".format(prefix,height,wrf_datetimes[dt_idx])
                fPath   = os.path.join(outPath,fName)
                print ("Saving : {0}".format(fName))
                dataset.to_netcdf(fPath)




            
