%% Nor_Smooth_Detrend_Regress_Filter pipeline
clc
clear
% psom_gb_vars
% Pipeline_opt.mode = 'qsub';
% Pipeline_opt.qsub_options = '-q long -l nodes=1:ppn=2';
% Pipeline_opt.mode_pipeline_manager = 'batch';
% Pipeline_opt.max_queued = 1100;
% Pipeline_opt.flag_verbose = 0;
% Pipeline_opt.flag_pause = 0;
% Pipeline_opt.path_logs = '/brain/gonggllab/GSP/Logs/Run2Step48_t4';
InputFiles=g_ls('/brain/gonggllab/GSP/Run2/FunImg/*/ranSub*.nii');
SubNum=length(InputFiles);
FFFiles=g_ls('/brain/gonggllab/GSP/Run2/T1Img/*/u_rc1co*_Scan_01_ANAT1_Template.nii');
DTFile='/brain/gonggllab/GSP/Run2/T1Img/Sub0001_Ses1/Template_6.nii';
BBox=[-90,-126,-72;90,90,108];
VoxSize=[3,3,3];
FWHM=[6 6 6];
PolyOrd=1;
GSMsk='/brain/gonggllab/Public/toolbox/GRETNA-2.0.0_release/Mask/BrainMask_3mm.nii';
WMMsk='/brain/gonggllab/Public/toolbox/GRETNA-2.0.0_release/Mask/WMMask_3mm.nii';
CSFMsk='/brain/gonggllab/Public/toolbox/GRETNA-2.0.0_release/Mask/CSFMask_3mm.nii';
HMInd=2;
HMFiles=g_ls('/brain/gonggllab/GSP/Run2/FunImg/*/HeadMotionParameter.txt');
TR=3;
Band=[0,0.08];
for i = 1:SubNum
    [a,prefix,~]=fileparts(InputFiles{i});
    FinishedFile=[a filesep 'bcWGSdsw' prefix '.nii'];
    if ~exist(FinishedFile,'file')
%         Job_Name = [ 'Run2_' num2str(i)];
%         Pipeline.(Job_Name).command = 'Nor_Smooth_Detrend_Regress_Filter(opt.parameters1,opt.parameters2,opt.parameters3,opt.parameters4,opt.parameters5,opt.parameters6,opt.parameters7,opt.parameters8,opt.parameters9,opt.parameters10,opt.parameters11,opt.parameters12,opt.parameters13,opt.parameters14)';
%         Pipeline.(Job_Name).opt.parameters1 = InputFiles(i);
%         Pipeline.(Job_Name).opt.parameters2 = FFFiles{i};
%         Pipeline.(Job_Name).opt.parameters3 = DTFile;
%         Pipeline.(Job_Name).opt.parameters4 = BBox;
%         Pipeline.(Job_Name).opt.parameters5 = VoxSize;
%         Pipeline.(Job_Name).opt.parameters6= FWHM;
%         Pipeline.(Job_Name).opt.parameters7 = PolyOrd;
%         Pipeline.(Job_Name).opt.parameters8 = GSMsk;
%         Pipeline.(Job_Name).opt.parameters9 = WMMsk;
%         Pipeline.(Job_Name).opt.parameters10 = CSFMsk;
%         Pipeline.(Job_Name).opt.parameters11 = HMInd;
%         Pipeline.(Job_Name).opt.parameters12= HMFiles(i);
%         Pipeline.(Job_Name).opt.parameters13 = TR;
%         Pipeline.(Job_Name).opt.parameters14= Band;
        Nor_Smooth_Detrend_Regress_Filter(InputFiles(i),FFFiles{i},DTFile,BBox,VoxSize,FWHM,PolyOrd,GSMsk,WMMsk,CSFMsk,HMInd,HMFiles(i),TR,Band);

    end
end
% psom_run_pipeline(Pipeline,Pipeline_opt);
