function RegisterfMRItoT1(InputFile)
[a,b,~]=fileparts(InputFile);
[aa,ID,~]=fileparts(a);
[aaa,~,~]=fileparts(aa);
fMRIOutpath=[a filesep 'fsl' ];
if ~exist(fMRIOutpath,'file')
    mkdir(fMRIOutpath);
end
T1ImgSourcepath=[aaa filesep 'T1Img' filesep ID];
T1ImgOutpath=[aaa filesep 'T1Img' filesep ID filesep 'fsl' filesep];
if ~exist(T1ImgOutpath,'file')
    mkdir(T1ImgOutpath);
end
GZfMRIFile=[fMRIOutpath filesep b '.nii.gz'];
cmd=['gzip -c ' InputFile ' > ' GZfMRIFile];
system(cmd);
OutfMRI=[fMRIOutpath filesep 'T1Space' b];
T1Img=[T1ImgSourcepath filesep 'co' ID '_Scan_01_ANAT1.nii'];
GZT1Img=[T1ImgOutpath filesep 'co' ID '_Scan_01_ANAT1.nii.gz'];
cmd=['gzip -c ' T1Img ' > ' GZT1Img];
system(cmd);
T1Brain=[T1ImgOutpath filesep 'bco' ID '_Scan_01_ANAT1.nii.gz'];
% BetBrainMask=[T1ImgOutpath filesep 'bco' ID '_Scan_01_ANAT1_mask.nii.gz'];
% cmd=['bet ' GZT1Img ' ' T1Brain ' -m '];
cmd=['bet ' GZT1Img ' ' T1Brain ];
system(cmd);
WMImg=[T1ImgSourcepath filesep 'c2co' ID '_Scan_01_ANAT1.nii'];
GZWMImg=[T1ImgOutpath filesep 'bco' ID '_Scan_01_ANAT1_wmseg.nii.gz'];
cmd=['gzip -c ' WMImg ' > ' GZWMImg];
system(cmd);
cmd=['epi_reg --wmseg=' GZWMImg ' --epi=' GZfMRIFile ' --t1=' GZT1Img ' --t1brain=' T1Brain ' --out=' OutfMRI];
system(cmd);
OutfMRI=[fMRIOutpath filesep 'T1Space' b '.nii.gz'];
cmd=['gunzip ' OutfMRI];
system(cmd);
