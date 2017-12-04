
% clc;
clear;
warning off
addpath('support_methods/');
addpath(genpath('tools'));
vl_setupnn;

Databases = {'ChallengeDB_release','LIVE','CSIQ','TID2013'};
trainDatabase = 'LIVE';
testDatabase = setdiff(Databases,trainDatabase); 

ModelType = 'AlexNet'; % ResNet  AlexNet S_CNN
trainingPropertion = 1.0;
testingPropertion = 1.0;
patchNum = [50];
%% parameters for PQR
quantizationMethod = 'uniform'; % LloydMax
beta = 64;
bins = 5;  % set bins=1 for scalar quality score regression

switch ModelType
    case 'AlexNet'
        patchSize = 227;
        patchStep = 16;
        epoch = 20;
    case 'ResNet'
        patchSize = 224;
        patchStep = 16;
        epoch = 10;
    case 'S_CNN'
        patchSize = 64;
        patchStep = 8;
        epoch = 40;
end

seed = 1; % no use in cross dataset eveluation

TrainModel('Model',ModelType,'Database',trainDatabase,'PatchNum', patchNum,... 
           'seed',seed, 'trainingpropertion', trainingPropertion,...
           'quantizationMethod',quantizationMethod,...
           'bins',bins, 'beta', beta,'patchSize',patchSize,...
           'patchStep',patchStep,'epoch',epoch);
for d = 1:numel(testDatabase)
    for i = 1:epoch
        netStruct = load(fullfile('data',[ModelType '_' trainDatabase '_TrainPropertion' num2str(trainingPropertion)...
                '_PatchSize' num2str(patchSize) '_PatchNum' num2str(patchNum) '_Seed' num2str(seed) '_bins' num2str(bins)...
                '_beta' num2str(beta)],['net-epoch-' num2str(i) '.mat']));
        net = dagnn.DagNN.loadobj(netStruct.net) ;
        move(net, 'gpu')
        net.mode = 'test';
        [SRCC(i),PLCC(i)] = testModel(ModelType, net, testDatabase{d},seed,1-testingPropertion);  
    end
    best_SRCC = max(abs(SRCC));
    best_PLCC = max((PLCC));
    file = fopen(fullfile('result','crossDatabase.txt'),'a');
    fprintf(file,'TrainDatabase: %s; TestDatabase: %s; Model: %s; bins = %d; patches = %d;\n',...
            trainDatabase,testDatabase{d},ModelType,bins,patchNum);
    fprintf(file,'SRCC: %.4f; PLCC: %.4f \n',best_SRCC,best_PLCC);
    fclose(file);
end

