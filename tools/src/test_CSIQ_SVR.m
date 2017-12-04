function test_CSIQ_SVR(testDatabase, testMethod, Trainingproportion)
    
    fprintf('Results on CSIQ:\n');
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    ref_list = dir(fullfile('databases', testDatabase,'src_imgs', '*.png'));
    
    mos = load(fullfile('databases', testDatabase, 'dmos.txt'));
    mos = mos';
    disClasses = dir(fullfile('databases', testDatabase,'dst_imgs'));
    disClasses = disClasses(3:end);
    count = 0;
    for type=1:length(disClasses)
        currentDisClassDirName = fullfile(testDatabase,'dst_imgs', disClasses(type,1).name);
        files = dir(fullfile('databases',currentDisClassDirName, '*.png'));
        for j=1:length(files)
            idx = strfind(files(j,1).name,'.');
            refImg = [files(j,1).name(1:idx(1)) 'png'];
            count = count + 1;
            disFilesList{count} = fullfile('databases',currentDisClassDirName, files(j,1).name);
            refFilesList{count} = fullfile('databases',testDatabase,'src_imgs', refImg);
            for p = 1:numel(ref_list)
                if strcmp(refImg,ref_list(p).name) == 1
                    classes(count) = p;
                    break;
                end
            end
        end
    end 
    
    if ~exist(fullfile('result', testDatabase, testMethod, 'features.mat'), 'file')
        tic;
        for i = 1:numel(disFilesList)
            fprintf('processing image %d / %d \n',i,numel(disFilesList));
            features{i} = CallingTheSourceCodeForFeatureExtraction(disFilesList{i},testMethod); 
        end
        features = cat(2,features{:});
        averageTime = toc/numel(disFilesList)
        save(fullfile('result', testDatabase, testMethod, 'features.mat'),'features','mos');
    else
        load(fullfile('result', testDatabase, testMethod, 'features.mat'),'features','mos');
    end
    
    idx = isnan(features);  features(idx) = 0; % for FRIQUEE
    
    tic;
    for seed = 1:10
        [trainingSet, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
        trainX = features(:,trainingSet)';
        testX = features(:,testingSet)';
        [trainX, testX] = featNormalize(trainX, testX);
        switch testMethod
            case 'BRISQUE'
                if Trainingproportion == 0.8
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 16 -g 0.05');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 16 -g 0.07');
                elseif Trainingproportion == 0.2
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 16 -g 0.09');
                end
                    
            case 'CORNIA'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 0');
            case 'HOSA'
                svm_model = train(mos(trainingSet)', sparse(trainX), '-s 11 -c 128');
            case 'DIIVINE'
                if Trainingproportion == 0.2
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 8 -g 0.03');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 8 -g 0.05');
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 8 -g 0.08');
                end
            case 'FRIQUEE'
                if Trainingproportion == 0.8
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 128 -g 0.02');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 128 -g 0.002');
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 128 -g 0.08');
                end
            otherwise 
                error('Unsupported Method!');
        end
        if strcmp(testMethod,'HOSA')
            [scores, ~,~] = predict(mos(testingSet)', sparse(testX), svm_model,'-q');
        else
            [scores, ~,~] = svmpredict(mos(testingSet)', double(testX), svm_model,'-q');
        end
        SRCC(seed) = corr(scores, mos(testingSet)', 'type', 'Spearman');
        PLCC(seed) = RegressionTID2013(scores, mos(testingSet)');
    end
    toc;
    fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));

end
