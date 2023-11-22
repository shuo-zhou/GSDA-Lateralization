function scrubbing_FC(InputFile,FDFile,ID,ScrubbingMethod,FDTrd,PreNum,PostNum,LabMask,OutputFC)
%% scrubbing
% [AllVolumeBrain, Header_Out] = y_Scrubbing(AllVolume, FDFile, ScrubbingMethod, PreNum,PostNum, Header)
% Do scrubbing
% Input:
% 	InputFile		-	the filename of one 4D data file
%   FDFile          -   the filename of FD file
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
OutputName=y_Scrubbing_adjusted(InputFile, FDFile, ScrubbingMethod,FDTrd,PreNum,PostNum);
%% FC
ZCX_fc(OutputName,LabMask,OutputFC,ID);
