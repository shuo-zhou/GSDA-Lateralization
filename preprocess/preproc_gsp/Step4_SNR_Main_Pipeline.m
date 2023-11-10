clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q fat8 -l nodes=1:ppn=2';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/jiangya/GSP/Logs/Run1SNR2';
InputFiles=g_ls('/brain/gonggllab/GSP/Run1/FunImg/*/fsl/T1Spaceran*_Scan_02_BOLD1.nii');
SubNum=length(InputFiles);
T1SourcePath='/brain/gonggllab/GSP/Run1/T1Img';
Run1SNR_native=cell(SubNum,2);
for i = 1:SubNum
    [a,b,~]=fileparts(InputFiles{i});
    FinishedFile=[a filesep 'SNR_native.mat'];
    if ~exist(FinishedFile,'file')
        Job_Name = ['Run1_' num2str(i)];
        Pipeline.(Job_Name).command = 'SNR_Main(opt.parameters1,opt.parameters2)';

        Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
        Pipeline.(Job_Name).opt.parameters2 = T1SourcePath;
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);


clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q fat8 -l nodes=1:ppn=2';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/jiangya/GSP/Logs/Run2SNR3';
InputFiles=g_ls('/brain/gonggllab/GSP/Run2/FunImg/*/fsl/T1Spaceran*_Scan_03_BOLD2.nii');
SubNum=length(InputFiles);
T1SourcePath='/brain/gonggllab/GSP/Run2/T1Img';
for i = 1:SubNum
    [a,b,~]=fileparts(InputFiles{i});
    FinishedFile=[a filesep 'SNR_native.mat'];
    if ~exist(FinishedFile,'file')
        Job_Name = ['Run2_' num2str(i)];
        Pipeline.(Job_Name).command = 'SNR_Main(opt.parameters1,opt.parameters2)';

        Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
        Pipeline.(Job_Name).opt.parameters2 = T1SourcePath;
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);
