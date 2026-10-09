%% readme notes with the ParWHData csv files
%
% fs: sampling frequency (Hz)
% amp: rms amplitudes of the multisine input u
% lines: frequency lines that are excited : 1 = DC
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ParWHData_Estimation_Level files: Estimation Multisines
% u: multisine estimation input dataset NP x M
% y: multisine estimation output dataset NP x M
%
% N = 16384; number of samples per period 
% P = 2; number of periods
% M = 20; number of random phase multisine realizations
% nAmp = 5; number of different amplitudes
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ParWHData_Validation_Level files: Validation Multisines
% u: multisine validation input dataset NP x 1
% y: multisine validation input dataset NP x 1
%
% N = 16384; number of samples per period 
% P = 2; number of periods
% M = 1; number of random phase multisine realizations
% nAmp = 5; number of different amplitudes
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ParWHData_ValidationArrow: Growing Amplitude Validation
% uValArr: growing amplitude gaussian noise validatioin NP x 1
% yValArr: growing amplitude gaussian noise validatioin NP x 1
%
% N = 16384; number of samples per period 
% P = 2; number of periods
%
% note: to compute the validation figures of merit on the ValArr signals a
% transient of 500 samples is removed first.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The data is transformed from the .mat format to .csv. A small loss in
% quality can have occured in this transformation. However, this loss of 
% quality is well below the noise level present in the data.