%% Rm_Sli_Reli pipeline
clc
clear
% psom_gb_vars
% Pipeline_opt.mode = 'qsub';
% Pipeline_opt.qsub_options = '-q short -l nodes=1:ppn=3';
% Pipeline_opt.mode_pipeline_manager = 'batch';
% Pipeline_opt.max_queued = 1100;
% Pipeline_opt.flag_verbose = 0;
% Pipeline_opt.flag_pause = 0;
% Pipeline_opt.path_logs = '/brain/gonggllab/GSP/Logs/Run2Step123_2';
RawFile=g_ls('/brain/gonggllab/GSP/Run2/FunImg/*/Sub*.nii');
SubNum=length(RawFile);
DelImg=4;
TR=3;
SInd=1;% alt+z
RInd=2;% Reference to middle
NumPasses=1;% Register to first
for i = 1:SubNum
    InputFile=RawFile(i);
    StrFile=RawFile{i};
    [a,b,~]=fileparts(StrFile);
    RANFile=[a filesep 'ran' b '.nii'];
    if ~exist(RANFile,'file')
        Rm_Sli_Reli(InputFile,DelImg,TR,SInd,RInd,NumPasses);

%         Job_Name = [ 'Run2_' num2str(i)];
%
%         Pipeline.(Job_Name).command = 'Rm_Sli_Reli(opt.parameters1,opt.parameters2,opt.parameters3,opt.parameters4,opt.parameters5,opt.parameters6)';
%         Pipeline.(Job_Name).opt.parameters1 = InputFile;
%         Pipeline.(Job_Name).opt.parameters2 = DelImg;
%         Pipeline.(Job_Name).opt.parameters3 = TR;
%         Pipeline.(Job_Name).opt.parameters4 = SInd;
%         Pipeline.(Job_Name).opt.parameters5 = RInd;
%         Pipeline.(Job_Name).opt.parameters6= NumPasses;
    end
end
% psom_run_pipeline(Pipeline,Pipeline_opt);
