function [SRCC,PLCC] = test_ChallengeDB(net,testDatabase, testMethod, Trainingproportion,seed)
    
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    fprintf('processing LIVE ChallengeDB images.....');
    Path = fullfile('databases', testDatabase);
    imageList = load(fullfile(Path,'Data','AllImages_release.mat'));
    mos = load(fullfile(Path,'Data','AllMOS_release.mat'));
    imageList = fullfile(Path,'Images',imageList.AllImages_release(8:end));
    mos = mos.AllMOS_release(8:end);
    
    if strcmp(testMethod,'CNN')
        trainingNum = ceil(length(mos)*Trainingproportion);
        rng(seed-1);
        randomsplit = randperm(length(mos),length(mos));
        testingSet = randomsplit(trainingNum+1:end);
        for i = 1:length(testingSet)
            scores(i) = processOneImage_CNN(net,imageList{testingSet(i)});
        end
        SRCC = corr(scores', mos(testingSet)', 'type', 'Spearman');
        PLCC = RegressionTID2013(scores', mos(testingSet)');
        fprintf('SRCC = %.4f;  PLCC = %.4f  \n', abs(SRCC), abs((PLCC)));
    else
        if ~exist(fullfile('result', testDatabase, testMethod, 'scores.mat'), 'file')
            for i = 1:length(imageList)
                fprintf('processing image %d / %d \n',i,numel(imageList));
                scores(i) = CallingTheSourceCodeForScorePrediction(imageList{i},testMethod);
            end
            save(fullfile('result', testDatabase, testMethod, 'scores.mat'),'scores');
        else
            load(fullfile('result', testDatabase, testMethod, 'scores.mat'),'scores');
        end

        trainingNum = ceil(length(mos)*Trainingproportion);
        for seed = 1:10
            rng(seed-1);
            randomsplit = randperm(length(mos),length(mos));
            testingSet = randomsplit(trainingNum+1:end);
            SRCC(seed) = corr(scores(testingSet)', mos(testingSet)', 'type', 'Spearman');
            PLCC(seed) = RegressionLIVE(scores(testingSet)', mos(testingSet)');
        end
        fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));
    end
end
