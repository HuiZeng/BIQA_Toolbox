function test_ChallengeDB_SVR(testDatabase, testMethod, Trainingproportion)
    
    %% set up database
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    fprintf('processing LIVE ChallengeDB images.....');
    Path = fullfile('databases', testDatabase);
    imageList = load(fullfile(Path,'Data','AllImages_release.mat'));
    mos = load(fullfile(Path,'Data','AllMOS_release.mat'));
    imageList = fullfile(Path,'Images',imageList.AllImages_release(8:end));
    mos = mos.AllMOS_release(8:end);
    
    
    %% feature extraction
    if ~exist(fullfile('result', testDatabase, testMethod, 'features.mat'), 'file')
        for i = 1:length(mos)
            fprintf('processing image %d / %d \n',i,length(mos));
            features{i} = CallingTheSourceCodeForFeatureExtraction(imageList{i},testMethod);
        end
        features = cat(2,features{:});
        save(fullfile('result', testDatabase, testMethod, 'features.mat'),'features');
    else
        load(fullfile('result', testDatabase, testMethod, 'features.mat'),'features');
    end
    
    %% train SVR and test
    tic;
    trainingNum = ceil(length(mos)*Trainingproportion);
    for seed = 1:10
        rng(seed-1);
        randomsplit = randperm(length(mos),length(mos));
        trainingSet = randomsplit(1:trainingNum);
        testingSet = randomsplit(trainingNum+1:end);
        trainX = features(:,trainingSet)';
        testX = features(:,testingSet)';
        [trainX, testX] = featNormalize(trainX, testX);
        switch testMethod
            case 'BRISQUE'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 10000 -g 0.05');
            case 'CORNIA'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 0');
            case 'HOSA'
                svm_model = train(mos(trainingSet)', sparse(trainX), '-s 11 -c 128');
            case 'DIIVINE'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 4096 -g 0.02');
            case 'FRIQUEE'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 256 -g 0.02');
            otherwise 
                error('Unsupported Method!');
        end
        if strcmp(testMethod,'HOSA')
            [scores, ~,~] = predict(mos(testingSet)', sparse(testX), svm_model,'-q');
        else
            [scores, ~,~] = svmpredict(mos(testingSet)', double(testX), svm_model,'-q');
        end
        
        SRCC(seed) = corr(scores, mos(testingSet)', 'type', 'Spearman');
        PLCC(seed) = RegressionLIVE(scores, mos(testingSet)');
    end
    toc;
    
    fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));

end
