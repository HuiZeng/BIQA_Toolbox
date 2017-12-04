function [SRCC,PLCC] = test_TID2013(net,testDatabase, testMethod, Trainingproportion,seed)
    
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    fprintf('Results on TID2013...');
    Path = fullfile('databases', testDatabase);
    type_scores = cell(24,1);
    type_mos = cell(24,1);
    
    
    fp = fopen(fullfile(Path,'mos_with_names.txt'),'r');
    for i = 1:3000
        temp = fgetl(fp);
        mos(i) = str2num(temp(1:7));
        classes(i) = str2num(temp(10:11));
        disImagePath{i} = fullfile(Path,'distorted_images',temp(9:end));
        type(i) = str2num(temp(13:14));
        if i <= 2880
            refImagePath{i} = fullfile(Path,'reference_images',['I' temp(10:11) '.BMP']);
        else
            refImagePath{i} = fullfile(Path,'reference_images',['i' temp(10:11) '.bmp']);
        end
    end
    fclose(fp);

    if strcmp(testMethod,'CNN')
        [~, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
        for i = 1:numel(testingSet)
            scores(i) = processOneImage_CNN(net,disImagePath{testingSet(i)});
        end
        SRCC = corr(scores', mos(testingSet)', 'type', 'Spearman');
        PLCC = RegressionTID2013(scores', mos(testingSet)');
        fprintf('SRCC = %.4f;  PLCC = %.4f  \n', abs(SRCC), abs(PLCC))
    else
        if ~exist(fullfile('result', testDatabase, testMethod, 'scores.mat'), 'file')
            for i = 1:numel(disImagePath)
                fprintf('processing image %d / %d \n',i,numel(disImagePath));
                scores(i) = CallingTheSourceCodeForScorePrediction(disImagePath{i},testMethod);
            end
            save(fullfile('result', testDatabase, testMethod, 'scores.mat'),'scores');
        else
            load(fullfile('result', testDatabase, testMethod, 'scores.mat'),'scores');
        end
        for seed = 1:10
            [~, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
            SRCC(seed) = corr(scores(testingSet)', mos(testingSet)', 'type', 'Spearman');
            PLCC(seed) = RegressionTID2013(scores(testingSet)', mos(testingSet)');
        end
        fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));
    end


end
