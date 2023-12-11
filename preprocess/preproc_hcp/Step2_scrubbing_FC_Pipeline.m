clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q fat8 -l nodes=1:ppn=4';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/HCP_PROCESS/Logs/R1XFC_AICHA0317';
InputFiles=g_ls('/brain/gonggllab/HCP_PROCESS/REST1/*/FunImg/*/bc*_hp2000_clean.nii');
% InputFiles=g_ls('/brain/gonggllab/HCP_PROCESS/REST1/LR/FunImg/100206/bcNGSd100206_rfMRI_REST1_LR_hp2000_clean.nii');
Num=length(InputFiles);
LabMask='/brain/gonggllab/HCP_PROCESS/AICHA_probability_tri_02_removed.nii';
if strfind(LabMask,'AICHA')
    MaskLabel='AICHA';
else strfind(LabMask,'BNA')
    MaskLabel='BNA';
end
ScrubbingMethod='cut';FDTrd=0.5;PreNum=1;PostNum=2;
for i = 1:Num
    [a,b,~]=fileparts(InputFiles{i});
    [aa,ID,~]=fileparts(a);
    [aaa,~,~]=fileparts(aa);
    FDFile=[aaa filesep 'RealignParameter' filesep ID filesep ID '_PowerFD.txt'];
    if ~exist(FDFile,'file')
        error('Can not find FDFile');
    end
    if strfind(b,'NGS')
        GlobalSignalLabel='NGR_scrub';
    else strfind(b,'WGS')
        GlobalSignalLabel='GR_scrub';
    end
    OutputFC=[aaa filesep GlobalSignalLabel filesep MaskLabel filesep];
    FinishedFile=[a filesep 'x' b '.nii'];
    if ~exist(FinishedFile,'file')
        Job_Name = ['R1AICHA_' num2str(i)];
        Pipeline.(Job_Name).command = 'scrubbing_FC(opt.parameters1,opt.parameters2,opt.parameters3,opt.parameters4,opt.parameters5,opt.parameters6,opt.parameters7,opt.parameters8,opt.parameters9)';
        Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
        Pipeline.(Job_Name).opt.parameters2 = FDFile;
        Pipeline.(Job_Name).opt.parameters3 = ID;
        Pipeline.(Job_Name).opt.parameters4 = ScrubbingMethod;
        Pipeline.(Job_Name).opt.parameters5 = FDTrd;
        Pipeline.(Job_Name).opt.parameters6 = PreNum;
        Pipeline.(Job_Name).opt.parameters7 = PostNum;
        Pipeline.(Job_Name).opt.parameters8 = LabMask;
        Pipeline.(Job_Name).opt.parameters9 = OutputFC;
        % scrubbing_FC(InputFiles{i},FDFile,ID,ScrubbingMethod,FDTrd,PreNum,PostNum,LabMask,OutputFC)
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);




clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q short -l nodes=1:ppn=6';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/HCP_PROCESS/Logs/R2XFC_AICHA0317';
InputFiles=g_ls('/brain/gonggllab/HCP_PROCESS/REST2/*/FunImg/*/bc*_hp2000_clean.nii');
Num=length(InputFiles);
LabMask='/brain/gonggllab/HCP_PROCESS/AICHA_probability_tri_02_removed.nii';
if strfind(LabMask,'AICHA')
    MaskLabel='AICHA';
else strfind(LabMask,'BNA')
    MaskLabel='BNA';
end
ScrubbingMethod='cut';FDTrd=0.5;PreNum=1;PostNum=2;
for i = 1:Num
    [a,b,~]=fileparts(InputFiles{i});
    [aa,ID,~]=fileparts(a);
    [aaa,~,~]=fileparts(aa);
    FDFile=[aaa filesep 'RealignParameter' filesep ID filesep ID '_PowerFD.txt'];
    if ~exist(FDFile,'file')
        error('Can not find FDFile');
    end
    if strfind(b,'NGS')
        GlobalSignalLabel='NGR_scrub';
    else strfind(b,'WGS')
        GlobalSignalLabel='GR_scrub';
    end
    OutputFC=[aaa filesep GlobalSignalLabel filesep MaskLabel filesep];
    FinishedFile=[a filesep 'x' b '.nii'];
    if ~exist(FinishedFile,'file')
        Job_Name = ['R2AICHA_' num2str(i)];
        Pipeline.(Job_Name).command = 'scrubbing_FC(opt.parameters1,opt.parameters2,opt.parameters3,opt.parameters4,opt.parameters5,opt.parameters6,opt.parameters7,opt.parameters8,opt.parameters9)';
        Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
        Pipeline.(Job_Name).opt.parameters2 = FDFile;
        Pipeline.(Job_Name).opt.parameters3 = ID;
        Pipeline.(Job_Name).opt.parameters4 = ScrubbingMethod;
        Pipeline.(Job_Name).opt.parameters5 = FDTrd;
        Pipeline.(Job_Name).opt.parameters6 = PreNum;
        Pipeline.(Job_Name).opt.parameters7 = PostNum;
        Pipeline.(Job_Name).opt.parameters8 = LabMask;
        Pipeline.(Job_Name).opt.parameters9 = OutputFC;
        % scrubbing_FC(InputFiles{i},FDFile,ID,ScrubbingMethod,FDTrd,PreNum,PostNum,LabMask,OutputFC)
    end
end
psom_run_pipeline(Pipeline,Pipeline_opt);



clc
clear
psom_gb_vars
Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q short -l nodes=1:ppn=4';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 100;
Pipeline_opt.flag_verbose = 0;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.path_logs = '/brain/gonggllab/HCP_PROCESS/Logs/rsfMRI_BNA';
InputFiles=g_ls('/brain/gonggllab/HCP_PROCESS/REST*/*/FunImg/*/xbc*.nii');
Num=length(InputFiles);
LabMask='/brain/gonggllab/HCP_PROCESS/BNA_probability_2mm_tri_02.nii';
if strfind(LabMask,'AICHA')
    MaskLabel='AICHA';
else strfind(LabMask,'BNA')
    MaskLabel='BNA';
end
for i = 1:Num
    [a,b,~]=fileparts(InputFiles{i});
    [aa,ID,~]=fileparts(a);
    [aaa,~,~]=fileparts(aa);
    if strfind(b,'NGS')
        GlobalSignalLabel='NGR_scrub';
    else strfind(b,'WGS')
        GlobalSignalLabel='GR_scrub';
    end
    OutputName=[aaa filesep GlobalSignalLabel filesep MaskLabel filesep];
    Job_Name = ['BNA_' num2str(i)];
    Pipeline.(Job_Name).command = 'ZCX_fc(opt.parameters1,opt.parameters2,opt.parameters3,opt.parameters4)';
    Pipeline.(Job_Name).opt.parameters1 = InputFiles{i};
    Pipeline.(Job_Name).opt.parameters2 = LabMask;
    Pipeline.(Job_Name).opt.parameters3 = OutputName;
    Pipeline.(Job_Name).opt.parameters4 = ID;
end
psom_run_pipeline(Pipeline,Pipeline_opt);



FDFiles=g_ls('/brain/gonggllab/HCP_PROCESS/REST2/*/RealignParameter/*/*_PowerFD.txt');
n=length(FDFiles);
PreNum=1;PostNum=2;TP=1200;FDTrd=0.5;
for i=1:n
    % FD Mask
    [Path,~,~]=fileparts(FDFiles{i});
    FD=load(FDFiles{i});

    FDMsk=FD>FDTrd;
    FDInd=find(FDMsk);
    PreMsk=false(TP, 1);
    for p=1:PreNum
        PreInd=FDInd-p;
        PreInd(PreInd<1)=[];
        PreMsk(PreInd)=true;
    end

    PostMsk=false(TP, 1);
    for j=1:PostNum
        PostInd=FDInd+j;
        PostInd(PostInd>TP)=[];
        PostMsk(PostInd)=true;
    end

    FDMsk=FDMsk | PreMsk | PostMsk;
    TempMask=~FDMsk;
    % Scrubbing Perctage
    SPFile=fullfile(Path, 'ScrubbingPerctage.txt');
    ScrubPerc=length(find(FDMsk))/TP;
    save(SPFile, 'ScrubPerc', '-ASCII', '-DOUBLE','-TABS');
    SMFile=fullfile(Path, 'ScrubbingMask.txt');
    FDMsk_Double=double(FDMsk);
    save(SMFile, 'FDMsk_Double', '-ASCII', '-DOUBLE','-TABS');
end
