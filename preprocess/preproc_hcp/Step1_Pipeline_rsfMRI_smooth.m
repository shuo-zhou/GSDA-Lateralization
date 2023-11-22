%Pipeline  Regression_Filter  For  rfMRI
clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q long -l nodes=1:ppn=12';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 50;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/HCP_PROCESS/Logs/200924sHCP_LR';


RawFileFolder=g_ls('/brain/gonggllab/HCP_PROCESS/REST1/LR/FunImg/*');
n=length(RawFileFolder);

% Smooth
FWHM=[4 4 4];
%Detrend parameter
PolyOrd=1;
%Regression parameter
GSMsk='/brain/gonggllab/HCP_PROCESS/Masks/AllResampled_BrainMask_05_91x109x91.nii';
WMMsk='/brain/gonggllab/HCP_PROCESS/Masks/AllResampled_WhiteMask_09_91x109x91.nii';
CSFMsk='/brain/gonggllab/HCP_PROCESS/Masks/AllResampled_CsfMask_07_91x109x91.nii';
%Filter parameter
TR=0.72;
FreBand=[0.01,0.1];%For rfMRI data

for i = 1:n
    [a,ID,~]=fileparts(RawFileFolder{i});
    [aa,~,~]=fileparts(a);
    [aaa,SessionLabel,~]=fileparts(aa);
    [~,RunLabel,~]=fileparts(aaa);
    InputFile=cellstr([RawFileFolder{i} filesep ID '_rfMRI_' RunLabel '_' SessionLabel '_hp2000_clean.nii']);
    FinishedFile=[RawFileFolder{i} filesep 'bcWGSds' ID '_rfMRI_' RunLabel '_' SessionLabel '_hp2000_clean.nii'];
    if ~exist(FinishedFile,'file')
        Job_Name = [ 's' num2str(i)];
        Pipeline.(Job_Name).command = 'rfMRI_preprocess_smooth(opt.parameters1,opt.parameters2,opt.parameters3,opt.parameters4,opt.parameters5,opt.parameters6,opt.parameters7,opt.parameters8)';
        Pipeline.(Job_Name).opt.parameters1 = InputFile;
        Pipeline.(Job_Name).opt.parameters2 = FWHM;
        Pipeline.(Job_Name).opt.parameters3 = PolyOrd;
        Pipeline.(Job_Name).opt.parameters4 = GSMsk;
        Pipeline.(Job_Name).opt.parameters5 = WMMsk;
        Pipeline.(Job_Name).opt.parameters6 = CSFMsk;
        Pipeline.(Job_Name).opt.parameters7 = TR;
        Pipeline.(Job_Name).opt.parameters8 = FreBand;
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);
