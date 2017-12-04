function [SRCC,PLCC] = test_CSIQ(net,testDatabase, testMethod, Trainingproportion,seed)
    
    fprintf('Results on CSIQ...');
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    ref_list = dir(fullfile('databases', testDatabase,'src_imgs', '*.png'));
    dmos = load(fullfile('databases', testDatabase, 'dmos.txt'));
    disClasses = dir(fullfile('databases', testDatabase,'dst_imgs'));
    count = 0;
    for i=1:length(disClasses)
        if strcmp(disClasses(i,1).name,'.') || strcmp(disClasses(i,1).name,'..')
            continue;
        end
        currentDisClassDirName = fullfile(testDatabase,'dst_imgs', disClasses(i,1).name);
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
    
    if strcmp(testMethod,'CNN')
        [~, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
        for i = 1:numel(testingSet)
            scores(i) = processOneImage_CNN(net,disFilesList{testingSet(i)}); 
        end
        SRCC = corr(scores', dmos(testingSet), 'type', 'Spearman');
        PLCC = RegressionCSIQ(scores', dmos(testingSet));
        fprintf('SRCC = %.4f;  PLCC = %.4f  \n', SRCC, PLCC);

    else
        if ~exist(fullfile('result', testDatabase, testMethod, 'scores.mat'), 'file')
            for i = 1:numel(disFilesList)
                fprintf('processing image %d / %d \n',i,numel(disFilesList));
                scores(i) = CallingTheSourceCodeForScorePrediction(disFilesList{i},testMethod);
            end
            save(fullfile('result', testDatabase, testMethod, 'scores.mat'),'scores');
        else
            load(fullfile('result', testDatabase, testMethod, 'scores.mat'),'scores');
        end

        for seed = 1:10
            [~, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
            SRCC(seed) = corr(scores(testingSet)', dmos(testingSet), 'type', 'Spearman');
            PLCC(seed) = RegressionCSIQ(scores(testingSet)', dmos(testingSet));
        end

        fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));
    end


end
