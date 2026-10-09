%% readme notes with ParWHData.mat
%
% fs: sampling frequency (Hz)
% amp: rms amplitudes of the multisine inputs (uEst,uVal)
% lines: frequency lines that are excited - 1 = DC
% uEst: multisine estimation input dataset N x P x M x nAmp
% yEst: multisine estimation output dataset N x P x M x nAmp
%
% N = 16384; number of samples per period 
% P = 2; number of periods
% M = 20; number of random phase multisine realizations
% nAmp = 5; number of different amplitudes
%
% uVal: multisine validation input dataset N x P x M x nAmp
% yVal: multisine validation input dataset N x P x M x nAmp
%
% N = 16384; number of samples per period 
% P = 2; number of periods
% M = 1; number of random phase multisine realizations
% nAmp = 5; number of different amplitudes
%
% uValArr: growing amplitude gaussian noise validatioin N x P
% yValArr: growing amplitude gaussian noise validatioin N x P
%
% N = 16384; number of samples per period 
% P = 2; number of periods
%
% note: to compute the validation figures of merit on the ValArr signals a
% transient of 500 samples is removed first.