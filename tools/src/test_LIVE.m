function [SRCC,PLCC] = test_LIVE(net,testDatabase, testMethod, Trainingproportion,seed)
    
    fprintf('Results on LIVE...');
    
    idx_jp2k = 1:227;
    idx_jpeg = max(idx_jp2k)+1:max(idx_jp2k)+233;
    idx_wn = max(idx_jpeg)+1:max(idx_jpeg)+174;
    idx_gblur = max(idx_wn)+1:max(idx_wn)+174;
    idx_fastfading = max(idx_gblur)+1:max(idx_gblur)+174;
    
    ref_list = dir(fullfile('databases', testDatabase,'refimgs', '*.bmp'));
    
    ref_names = load(fullfile('databases', testDatabase, 'refnames_all.mat'));
    ref_names = ref_names.refnames_all;
    for i = 1:numel(ref_names)
        for j = 1:numel(ref_list)
            if strcmp(ref_names{i},ref_list(j).name) == 1
                classes(i) = j;
                break;
            end
        end
    end

    refImagePath = {};
    for index = 1:982
        refPath = fullfile('databases', testDatabase, 'refimgs', char(ref_names(index)));
        refImagePath{index} = refPath;
    end
    
    load(fullfile('databases', testDatabase, 'dmos_realigned.mat'));
    dmos = dmos_new;
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    count = 1;
    for i = 1:numel(idx_jp2k)
        disImgPath{count} = fullfile('databases', testDatabase,'jp2k', ['img' num2str(i) '.bmp']);
        count = count + 1;
    end
    for i = 1:numel(idx_jpeg)
        disImgPath{count} = fullfile('databases', testDatabase,'jpeg', ['img' num2str(i) '.bmp']);
        count = count + 1;
    end
    for i = 1:numel(idx_wn)
        disImgPath{count} = fullfile('databases', testDatabase,'wn', ['img' num2str(i) '.bmp']);
        count = count + 1;
    end
    for i = 1:numel(idx_gblur)
        disImgPath{count} = fullfile('databases', testDatabase,'gblur', ['img' num2str(i) '.bmp']);
        count = count + 1;
    end
    for i = 1:numel(idx_fastfading)
        disImgPath{count} = fullfile('databases', testDatabase,'fastfading', ['img' num2str(i) '.bmp']);
        count = count + 1;
    end
    
    sel = find(orgs == 0);
    dmos = dmos(sel);
    classes = classes(sel);
    refImagePath = refImagePath(sel);
    disImgPath = disImgPath(sel);
    
    if strcmp(testMethod,'CNN')
        [~, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
        for i = 1:numel(testingSet)
            scores(i) = processOneImage_CNN(net,disImgPath{testingSet(i)});
        end
        SRCC = corr(scores', dmos(testingSet)', 'type', 'Spearman');
        PLCC = RegressionLIVE(scores', dmos(testingSet)');
        fprintf('SRCC = %.4f;  PLCC = %.4f  \n', abs(SRCC), abs(PLCC));
    else
        if ~exist(fullfile('result', testDatabase, testMethod, 'scores.mat'), 'file')
            for i=1:length(disImgPath)
                fprintf('processing image %d / %d \n',i,numel(disImgPath));
                scores(i) = CallingTheSourceCodeForScorePrediction(disImgPath{i},testMethod);
            end
            save(fullfile('result', testDatabase, testMethod,'scores.mat'),'scores');
        else
            load(fullfile('result', testDatabase, testMethod,'scores.mat'),'scores');
        end

        for seed = 1:10
            [~, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
            SRCC(seed) = corr(scores(testingSet)', dmos(testingSet)', 'type', 'Spearman');
            PLCC(seed) = RegressionLIVE(scores(testingSet)', dmos(testingSet)');
        end

        fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));
    end
end
