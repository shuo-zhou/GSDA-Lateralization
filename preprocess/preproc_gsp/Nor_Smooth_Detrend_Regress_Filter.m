function Nor_Smooth_Detrend_Regress_Filter(InputFile, FFFile, DTFile, BBox, VoxSize,FWHM,PolyOrd,GSMsk, WMMsk, CSFMsk, HMInd, HMFile,TR, Band)
[a,prefix,~]=fileparts(InputFile{1});
gretna_RUN_DartelNormEpi(InputFile, FFFile, DTFile, BBox, VoxSize);
NormalizedFile={[a filesep 'w' prefix '.nii']};
gretna_RUN_Smooth(NormalizedFile, FWHM);
SmoothFile={[a filesep 'sw' prefix '.nii']};
gretna_RUN_Detrend(SmoothFile, PolyOrd);
DetrendFile={[a filesep 'dsw' prefix '.nii']};
gretna_RUN_RegressOut(DetrendFile, GSMsk, WMMsk, CSFMsk, HMInd, HMFile);
RegressFile={[a filesep 'cWGSdsw' prefix '.nii']};
gretna_RUN_Filter(RegressFile, TR, Band);
