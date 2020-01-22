clear all, close all, clc

path_v={'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_12_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_14_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_15_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_18_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_19_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_20_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_27_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_28_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_29_mesoLES/',...
'/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/WRF_mesoLES_4dom_RAP_2015_03_30_mesoLES/'};
path_v=path_v{1};
hour_v=14:23;

for p_f=1:1%length(path_v)
  path=path_v%{p_f}
  fflush(stdout);
  cd(path)
  ust=0;
  for hh=1:length(hour_v)
     for ff=1:2
        folder_res=['HOUR_',num2str(hour_v(hh)),'_',num2str(ff)]
        fflush(stdout);
        cd(folder_res)
        a='wrfout'
        fflush(stdout);
        name_list=dir('wrfout*');
        length(name_list)
        fflush(stdout);
        for pp=1:length(name_list)
          file_i=['name_list(',num2str(pp),').name'];      
          file_i=eval(file_i);
          ust_i=mean(mean(mean(ncread(file_i,'UST'))));
          ust=ust+ust_i;
        end
        cd ..
     end
  end
end

cd '/glade2/scratch2/domingom/Cheyenne/XPIA_mesoLES/SIMULS/'

save('Mean_ust_out_01.mat','ust','-mat7-binary')

aaa='SUCCESSFULLY COMPLETED'
fflush(stdout);
