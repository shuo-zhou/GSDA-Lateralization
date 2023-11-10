function rfMRI_preprocess_smooth(InputFile,FWHM,PolyOrd,GSMsk, WMMsk, CSFMsk, TR,FreBand)
% Smooth
S_File=gretna_RUN_Smooth(InputFile, FWHM);
%Detrend
D_File=gretna_RUN_Detrend(S_File, PolyOrd);
%Regression   NoGlobalSignal
NGR_File=gretna_RUN_RegressOut(D_File, [], WMMsk, CSFMsk, 0, []);
%Regression   GlobalSignal
GR_File=gretna_RUN_RegressOut(D_File, GSMsk, WMMsk, CSFMsk, 0, []);
%Filter
gretna_RUN_Filter(NGR_File, TR, FreBand);
gretna_RUN_Filter(GR_File, TR, FreBand);
