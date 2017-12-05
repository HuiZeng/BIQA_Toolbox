
% clc;
clear;
warning off
addpath('support_methods/');
addpath(genpath('tools'));
vl_setupnn;

%% general parameters
% 4 supported datasets: ChallengeDB_release LIVE TID2013 CSIQ
testDatabase = 'ChallengeDB_release'; 
% 3 supported CNN architectures: ResNet AlexNet S_CNN
ModelType = 'S_CNN'; 
repetitions = 10;
trainingPropertion = 0.8;
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

for patchNum = [50]
    if strcmp(ModelType,'S_CNN')
        patchNum = patchNum * 10;
    end

%% training
for seed = 1:repetitions
    TrainModel('Model',ModelType,'Database',testDatabase,'PatchNum', patchNum,... 
               'seed',seed, 'trainingpropertion', trainingPropertion,...
               'quantizationMethod',quantizationMethod,...
               'bins',bins, 'beta', beta,'patchSize',patchSize,...
               'patchStep',patchStep,'epoch',epoch);
end

%% testing
for seed = 1:repetitions
    fprintf('seed = %d....\n', seed);
    for i = 1:epoch
        netStruct = load(fullfile('data',[ModelType '_' testDatabase '_TrainPropertion' num2str(trainingPropertion)...
                    '_PatchSize' num2str(patchSize) '_PatchNum' num2str(patchNum) '_Seed' num2str(seed) '_bins' num2str(bins)...
                    '_beta' num2str(beta)],['net-epoch-' num2str(i) '.mat']));
        net = dagnn.DagNN.loadobj(netStruct.net) ;
        move(net, 'gpu')
        net.mode = 'test';
        [SRCC(seed,i),PLCC(seed,i)] = testModel(ModelType, net, testDatabase,seed,trainingPropertion); 
    end
end

%% save results
median_SRCC =[]; median_PLCC=[]; std_SRCC=[]; std_PLCC=[];
median_SRCC = median(SRCC);
median_PLCC = median(PLCC);
std_SRCC = std(SRCC);
std_PLCC = std(PLCC);
[max_SRCC,idx] = max(median_SRCC);
max_PLCC = median_PLCC(idx);
file = fopen(fullfile('result','results.txt'),'a');
fprintf(file,'Parameters: %s; %s; bins = %d; patches = %d;\n',testDatabase,ModelType,bins,patchNum);
fprintf(file,'SRCC:     ');fprintf(file,'%.4f; ',median_SRCC);fprintf(file,'\n');
fprintf(file,'std_SRCC: ');fprintf(file,'%.4f; ',std_SRCC);fprintf(file,'\n');
fprintf(file,'PLCC:     ');fprintf(file,'%.4f; ',median_PLCC);fprintf(file,'\n');
fprintf(file,'std_PLCC: ');fprintf(file,'%.4f; ',std_PLCC);fprintf(file,'\n');
fprintf(file,'best_SRCC: %.4f; best_PLCC: %.4f \n',max_SRCC,max_PLCC);
fclose(file);

end
