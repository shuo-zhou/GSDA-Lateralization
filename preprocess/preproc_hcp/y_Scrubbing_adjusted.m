function [OutputName] = y_Scrubbing_adjusted(InputFile, FDFile, ScrubbingMethod,FDTrd,PreNum,PostNum)
% [AllVolumeBrain, Header_Out] = y_Scrubbing(AllVolume, FDFile, ScrubbingMethod, PreNum,PostNum, Header)
% Do scrubbing
% Input:
% 	InputFile		-	the filename of one 4D data file
%   FDFile          -   the filename of FD file
%   TemporalMask    -   Temporal mask for scrubbing (DimTimePoints*1)
%   ScrubbingMethod -   The methods for scrubbing.
%                       -1. 'cut': discarding the timepoints with TemporalMask == 0
%                       -2. 'nearest': interpolate the timepoints with TemporalMask == 0 by Nearest neighbor interpolation
%                       -3. 'linear': interpolate the timepoints with TemporalMask == 0 by Linear interpolation
%                       -4. 'spline': interpolate the timepoints with TemporalMask == 0 by Cubic spline interpolation
%                       -5. 'pchip': interpolate the timepoints with TemporalMask == 0 by Piecewise cubic Hermite interpolation
%   FDTrd      - FD Threshold
%   PreNum     - The time point number removed before bad time point
%   PostNum    - The time point number removed after bad time point
% Output:
%	AllVolumeBrain      -   The AllVolume after scrubbing
%   Header_Out          -   The NIfTI Header
%   All the volumes after scrubbing will be output as where OutputName specified.
%-----------------------------------------------------------
% Written by YAN Chao-Gan 120423.
% The Nathan Kline Institute for Psychiatric Research, 140 Old Orangeburg Road, Orangeburg, NY 10962, USA
% Child Mind Institute, 445 Park Avenue, New York, NY 10022, USA
% The Phyllis Green and Randolph Cowen Institute for Pediatric Neuroscience, New York University Child Study Center, New York, NY 10016, USA
% ycg.yan@gmail.com

[AllVolume,~,~, Header] =y_ReadAll(InputFile);
[nDim1,nDim2,nDim3,TP]=size(AllVolume);
[a,b,~]=fileparts(InputFile);
OutputName=[a filesep 'x' b '.nii'];

% Convert into 2D
AllVolume=reshape(AllVolume,[],TP)';
% FD Mask
[Path,~,~]=fileparts(FDFile);
FD=load(FDFile);

FDMsk=FD>FDTrd;
FDInd=find(FDMsk);
PreMsk=false(TP, 1);
for i=1:PreNum
    PreInd=FDInd-i;
    PreInd(PreInd<1)=[];
    PreMsk(PreInd)=true;
end

PostMsk=false(TP, 1);
for i=1:PostNum
    PostInd=FDInd+i;
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

% Scrubbing
if ~all(TempMask)
    AllVolume = AllVolume(find(TempMask),:); %'cut'
    if ~strcmpi(ScrubbingMethod,'cut')
        xi=1:length(TempMask);
        x=xi(find(TempMask));
        AllVolume = interp1(x,AllVolume,xi,ScrubbingMethod);
    end
    TP = size(AllVolume,1);
end


AllVolumeBrain = zeros(TP, nDim1*nDim2*nDim3);
AllVolumeBrain=AllVolume;
AllVolumeBrain=reshape(AllVolumeBrain',[nDim1, nDim2, nDim3, TP]);

Header_Out = Header;
Header_Out.pinfo = [1;0;0];
Header_Out.dt    = [16,0];
Header_Out.dim   = [nDim1, nDim2, nDim3, TP];

y_Write(AllVolumeBrain,Header_Out,OutputName);
