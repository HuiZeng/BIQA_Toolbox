function test_LIVE_SVR(testDatabase, testMethod, Trainingproportion)
    
    fprintf('Results on LIVE:\n');
    
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
    mos = dmos_new;
    if ~exist(fullfile('result', testDatabase, testMethod), 'dir')
        mkdir(fullfile('result', testDatabase, testMethod));
    end
    
    if ~exist(fullfile('result', testDatabase, testMethod, 'features.mat'), 'file')
        for i = 1:numel(idx_jp2k)
            disImgPath = fullfile('databases', testDatabase,'jp2k', ['img' num2str(i) '.bmp']);
            feat_jp2k{i} = CallingTheSourceCodeForFeatureExtraction(disImgPath,testMethod);
        end
        for i = 1:numel(idx_jpeg)
            disImgPath = fullfile('databases', testDatabase,'jpeg', ['img' num2str(i) '.bmp']);
            feat_jpeg{i} = CallingTheSourceCodeForFeatureExtraction(disImgPath,testMethod);
        end
        for i = 1:numel(idx_wn)
            disImgPath = fullfile('databases', testDatabase,'wn', ['img' num2str(i) '.bmp']);
            feat_wn{i} = CallingTheSourceCodeForFeatureExtraction(disImgPath,testMethod);
        end
        for i = 1:numel(idx_gblur)
            disImgPath = fullfile('databases', testDatabase,'gblur', ['img' num2str(i) '.bmp']);
            feat_gblur{i} = CallingTheSourceCodeForFeatureExtraction(disImgPath,testMethod);
        end
        for i = 1:numel(idx_fastfading)
            disImgPath = fullfile('databases', testDatabase,'fastfading', ['img' num2str(i) '.bmp']);
            feat_fastfading{i} = CallingTheSourceCodeForFeatureExtraction(disImgPath,testMethod);
        end
        features = cat(2,feat_jp2k{:},feat_jpeg{:},feat_wn{:},feat_gblur{:},feat_fastfading{:});
        sel = find(orgs == 0);
        features = features(:,sel);
        mos = mos(sel);
        classes = classes(sel);
        save(fullfile('result', testDatabase, testMethod,'features.mat'),'features','mos','classes');
    else
        load(fullfile('result', testDatabase, testMethod,'features.mat'),'features','mos','classes');
    end
    
    tic;
    for seed = 1:10
        [trainingSet, testingSet] = generateTrainingSet(classes,seed,Trainingproportion);
        trainX = features(:,trainingSet)';
        testX = features(:,testingSet)';
        [trainX, testX] = featNormalize(trainX, testX);

        switch testMethod
            case 'BRISQUE'
                if Trainingproportion == 0.2
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 100000 -g 0.01'); 
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 100000 -g 0.05'); 
                end
            case 'CORNIA'
                svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 0');
            case 'HOSA'
                svm_model = train(mos(trainingSet)', sparse(trainX), '-s 11 -c 128');
            case 'DIIVINE'
                if Trainingproportion == 0.8
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 10000 -g 0.1');
                elseif Trainingproportion == 0.5
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 10000 -g 0.04');
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 10000 -g 0.01');
                end
            case 'FRIQUEE'
                if Trainingproportion == 0.2
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 256 -g 0.01');
                else
                    svm_model = svmtrain(mos(trainingSet)', double(trainX), '-s 4 -t 2 -c 256 -g 0.03');
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
        PLCC(seed) = RegressionLIVE(scores, mos(testingSet)');
    end
    toc;
    fprintf('SRCC = %.4f, std = %.4f;  PLCC = %.4f, std = %.4f \n', abs(median(SRCC)), std(SRCC), abs(median((PLCC))), std((PLCC)));

end


