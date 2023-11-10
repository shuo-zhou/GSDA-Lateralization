%% RegisterfMRItoT1 pipeline
clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q fat8 -l nodes=1:ppn=2';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/jiangya/GSP/Logs/Run1Reg';
InputFiles=g_ls('/brain/gonggllab/GSP/Run1/FunImg/*/ranSub*.nii');
SubNum=length(InputFiles);
for i = 1:SubNum
    [a,b,~]=fileparts(InputFiles{i});
    FinishedFile=[a filesep 'fsl' filesep 'T1Space' b '.nii'];
    if ~exist(FinishedFile,'file')
        Job_Name = ['Run1_' num2str(i)];
        Pipeline.(Job_Name).command = 'RegisterfMRItoT1(opt.parameters1)';
        Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);

InputFiles=g_ls('/brain/gonggllab/GSP/Run1/FunImg/*/ranSub*.nii');
SubNum=length(InputFiles);
parfor i = 1:SubNum
    RegisterfMRItoT1(InputFiles{i});
    disp(['Sub' num2str(i) 'th has finished']);
end

clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q fat8 -l nodes=1:ppn=2';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/jiangya/GSP/Logs/Run2Reg2';
InputFiles=g_ls('/brain/gonggllab/GSP/Run2/FunImg/*/ranSub*.nii');
SubNum=length(InputFiles);
for i = 1:SubNum
    [a,b,~]=fileparts(InputFiles{i});
    FinishedFile=[a filesep 'fsl' filesep 'T1Space' b '.nii'];
    if ~exist(FinishedFile,'file')
        Job_Name = ['Run2_' num2str(i)];
        Pipeline.(Job_Name).command = 'RegisterfMRItoT1(opt.parameters1)';
        Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
        % RegisterfMRItoT1(InputFiles{i});
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);
