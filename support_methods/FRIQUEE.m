function [score] = FRIQUEE(I)

testFriqueeFeats = extractFRIQUEEFeatures(I);

% Load a learned model
load('support_functions/FRIQUEE/data/friqueeLearnedModel.mat');

% Scale the features of the test image accordingly.
% The minimum and the range are computed on features of all the images of
% LIVE Challenge Database
testFriqueeALL = testFeatNormalize(testFriqueeFeats.friqueeALL, friqueeLearnedModel.trainDataMinVals, friqueeLearnedModel.trainDataRange);

score = svmpredict (0, double(testFriqueeALL), friqueeLearnedModel.trainModel, '-b 1 -q');

end