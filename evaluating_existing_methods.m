% clc;
clear;
warning off;

addpath('support_methods/');
addpath(genpath('tools'));

testDatabase = {'ChallengeDB_release'}; % LIVE CSIQ TID2013 ChallengeDB_release
testMethod = {'NIQE'}; % BRISQUE, DIIVINE, FRIQUEE, CORNIA, NIQE, ILNIQE, HOSA
trainingPropertion = 0.8;
repetitions = 10;


for j = 1:numel(testMethod)
    addpath(genpath(fullfile('support_methods',testMethod{j})));
    if strcmp(testMethod{j},'NIQE') || strcmp(testMethod{j},'ILNIQE')
        for i = 1:numel(testDatabase)
            switch testDatabase{i}
                case 'LIVE'
                    test_LIVE([],testDatabase{i},testMethod{j}, trainingPropertion);
                case 'TID2013'
                    test_TID2013([],testDatabase{i},testMethod{j}, trainingPropertion);
                case 'CSIQ'
                    test_CSIQ([],testDatabase{i},testMethod{j}, trainingPropertion);
                case 'ChallengeDB_release'
                    test_ChallengeDB([],testDatabase{i},testMethod{j}, trainingPropertion);
                case 'LIVE_MD'
                    test_LIVE_MD([],testDatabase{i},testMethod{j}, trainingPropertion);
                otherwise
                    error('Unknown database!');
            end
        end
    else
        for i = 1:numel(testDatabase)
            switch testDatabase{i}
                case 'LIVE'
                    test_LIVE_SVR(testDatabase{i},testMethod{j}, trainingPropertion);
                case 'TID2013'
                    test_TID2013_SVR(testDatabase{i},testMethod{j}, trainingPropertion);
                case 'CSIQ'
                    test_CSIQ_SVR(testDatabase{i},testMethod{j}, trainingPropertion);
                case 'ChallengeDB_release'
                    test_ChallengeDB_SVR(testDatabase{i},testMethod{j}, trainingPropertion);
                case 'LIVE_MD'
                    test_LIVE_MD_SVR(testDatabase{i},testMethod{j}, trainingPropertion);
                otherwise
                    error('Unknown database!');
            end
        end
    end
    rmpath(genpath(fullfile('support_methods',testMethod{j})));
end
