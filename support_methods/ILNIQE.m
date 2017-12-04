function [score] = ILNIQE(I)

templateModel = load('templateModel.mat');
templateModel = templateModel.templateModel;
mu_prisparam = templateModel{1};
cov_prisparam = templateModel{2};
meanOfSampleData = templateModel{3};
principleVectors = templateModel{4};

score = computequality(I,mu_prisparam,cov_prisparam,principleVectors,meanOfSampleData);

end