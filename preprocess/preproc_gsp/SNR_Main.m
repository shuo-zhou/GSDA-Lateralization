function SNR_Main(InputFile,T1SourcePath)
[a,~,~]=fileparts(InputFile);
[aa,~,~]=fileparts(a);
[~,ID,~]=fileparts(aa);
SNR_native=cell(1,2);
SNR_native{1,1}=ID;

c1=[T1SourcePath filesep ID filesep 'c1co' ID '_Scan_01_ANAT1.nii'];
c1Mask=[T1SourcePath filesep ID filesep 'fsl' filesep 'mc1co' ID '_Scan_01_ANAT1.nii.gz'];
cmd=['fslmaths ' c1 ' -thr 0.25 -bin ' c1Mask];
system(cmd);
c2=[T1SourcePath filesep ID filesep 'c2co' ID '_Scan_01_ANAT1.nii'];
c2Mask=[T1SourcePath filesep ID filesep 'fsl' filesep 'mc2co' ID '_Scan_01_ANAT1.nii.gz'];
cmd=['fslmaths ' c2 ' -thr 0.5 -bin ' c2Mask];
system(cmd);
c1c2Mask=[T1SourcePath filesep ID filesep 'mc2c1co' ID '_Scan_01_ANAT1.nii.gz'];
cmd=['fslmaths ' c1Mask ' -add ' c2Mask ' -thr 0  -bin ' c1c2Mask];
system(cmd);
c3=[T1SourcePath filesep ID filesep 'c3co' ID '_Scan_01_ANAT1.nii'];
c3Mask=[T1SourcePath filesep ID filesep 'fsl' filesep 'mc3co' ID '_Scan_01_ANAT1.nii.gz'];
cmd=['fslmaths ' c3 ' -thr 0.5 -bin ' c3Mask];
system(cmd);
c1c2c3Mask=[T1SourcePath filesep ID filesep 'mc3c2c1co' ID '_Scan_01_ANAT1.nii.gz'];
cmd=['fslmaths ' c1c2Mask ' -add ' c3Mask ' -thr 0 -bin ' c1c2c3Mask];
system(cmd);
BrainMask=c1c2c3Mask;
SNR_native{1,2}=SignalToNoiseRatio(InputFile,BrainMask);
save([a filesep 'SNR_native.mat'],'SNR_native');
disp([ID ' has finished.'])
