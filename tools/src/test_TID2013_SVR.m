function test_TID2013_SVR(testDatabase, testMethod, Trainingproportion)
    
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    fprintf('processing TID2013 images.....');
    Path = fullfile('databases', testDatabase);
    type_scores = cell(24,1);
    type_mos = cell(24,1);
    
    if ~exist(fullfile('result', testDatabase, testMethod, 'features.mat'), 'file')
        fp = fopen(fullfile(Path,'mos_with_names.txt'),'r');
        for i = 1:3000
            fprintf('processing image %d / 3000 \n',i);
            temp = fgetl(fp);
            mos(i) = str2num(temp(1:7));
            classes(i) = str2num(temp(10:11));
            ImagePath = fullfile(Path,'distorted_images',temp(9:end));
            type = str2num(temp(13:14));
            if i <= 2880
                refImagePath = fullfile(Path,'reference_images',['I' temp(10:11) '.BMP']);
            else
                refImagePath = fullfile(Path,'reference_images',['i' temp(10:11) '.bmp']);
            end
            features{i} = CallingTheSourceCodeForFeatureExtraction(ImagePath,testMethod);
        end
        fclose(fp);
        features = cat(2,features{:});
        save(fullfile('result', testDatabase, testMethod, 'features.mat'),'features','mos','classes');
    else
        load(fullfile('result', testDatabase, testMethod, 'features.mat'),'features','mos','classes');
    end
    
    idx = isnan(features); features(idx) = 0;  % for FRIQUEE
    
    tic;
    for seed = 1:10
        [trainingSet, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
        trainX = features(:,trainingSet)';
        testX = features(:,testingSet)';
        
        if ~strcmp(testMethod,'HOSA')
            [trainX, testX] = featNormalize(trainX, testX);
        end
        
        switch testMethod
            case 'BRISQUE'
                if Trainingproportion == 0.2
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 256 -g 0.005');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 128 -g 0.05');
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 256 -g 0.005');
                end
            case 'CORNIA'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 0');
            case 'HOSA'
                if Trainingproportion == 0.8
                    svm_model = train(mos(trainingSet)', sparse(trainX), '-s 11 -c 1');
                else
                    svm_model = train(mos(trainingSet)', sparse(trainX), '-s 11 -c 4');
                end
            case 'DIIVINE'
                if Trainingproportion == 0.8
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 4 -g 0.5');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 4 -g 0.3');
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 4 -g 0.06');
                end
            case 'FRIQUEE'
                if Trainingproportion == 0.8
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 0.5 -g 0.05');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 1 -g 0.1');
                elseif Trainingproportion == 0.2
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 2 -g 0.05');
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
