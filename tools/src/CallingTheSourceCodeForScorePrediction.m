function score = CallingTheSourceCodeForScorePrediction(imagePath,testMethod)

I = imread(imagePath);

switch testMethod
    case 'NIQE'
        load modelparameters.mat
        blocksizerow    = 96;
        blocksizecol    = 96;
        blockrowoverlap = 0;
        blockcoloverlap = 0;
        score = computequality(I,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, mu_prisparam,cov_prisparam);
    case 'ILNIQE'
        templateModel = load('templateModel.mat');
        templateModel = templateModel.templateModel;
        mu_prisparam = templateModel{1};
        cov_prisparam = templateModel{2};
        meanOfSampleData = templateModel{3};
        principleVectors = templateModel{4};
        score = computequality(I,mu_prisparam,cov_prisparam,principleVectors,meanOfSampleData);
    otherwise
        error('Unsupported Method!');
end

end