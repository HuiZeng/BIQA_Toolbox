% --------------------------------------------------------------------
function imdb = getImdbLIVE(opts)
% --------------------------------------------------------------------

opts.dataDir = fullfile('databases', 'LIVE') ;
idx_jp2k = 1:227;
idx_jpeg = max(idx_jp2k)+1:max(idx_jp2k)+233;
idx_wn = max(idx_jpeg)+1:max(idx_jpeg)+174;
idx_gblur = max(idx_wn)+1:max(idx_wn)+174;
idx_fastfading = max(idx_gblur)+1:max(idx_gblur)+174;

ref_list = dir(fullfile('databases', 'LIVE','refimgs', '*.bmp'));
ref_names = load(fullfile('databases','LIVE','refnames_all.mat'));
ref_names = ref_names.refnames_all;
for i = 1:numel(ref_names)
    for j = 1:numel(ref_list)
        if strcmp(ref_names{i},ref_list(j).name) == 1
            classes(i) = j;
            break;
        end
    end
end
    
load(fullfile(opts.dataDir,'dmos_realigned.mat'));
dmos = dmos_new;
dmos = (dmos - min(dmos)) / (max(dmos) - min(dmos));
cnt = 0;
for i = 1:numel(idx_jp2k)
    cnt = cnt + 1;
    disImgPath{cnt} = fullfile(opts.dataDir,'jp2k', ['img' num2str(i) '.bmp']);
end
for i = 1:numel(idx_jpeg)
    cnt = cnt + 1;
    disImgPath{cnt} = fullfile(opts.dataDir,'jpeg', ['img' num2str(i) '.bmp']);
end
for i = 1:numel(idx_wn)
    cnt = cnt + 1;
    disImgPath{cnt} = fullfile(opts.dataDir,'wn', ['img' num2str(i) '.bmp']);
end
for i = 1:numel(idx_gblur)
    cnt = cnt + 1;
    disImgPath{cnt} = fullfile(opts.dataDir, 'gblur', ['img' num2str(i) '.bmp']);
end
for i = 1:numel(idx_fastfading)
    cnt = cnt + 1;
    disImgPath{cnt} = fullfile(opts.dataDir, 'fastfading', ['img' num2str(i) '.bmp']);
end
sel = orgs([idx_jp2k,idx_jpeg,idx_wn,idx_gblur,idx_fastfading]);
dmos = dmos(sel==0);
disImgPath = disImgPath(sel==0);
classes = classes(sel==0);
[trainingSet, testingSet] = generateTrainingSet(classes,opts.seed,opts.trainingpropertion);
dmos = dmos(trainingSet);
disImgPath = disImgPath(trainingSet);

for i = 1:numel(trainingSet)
    fprintf('Finished image %d / %d...\n', i, numel(trainingSet));
    I = imread(disImgPath{i});
    I = localNormalization(I,opts.filter_);
    data{i} = generatepatch(I,opts.PatchSize,opts.PatchNum,opts.PatchStep);
    score{i} = ones(1,size(data{i},4)) * dmos(i);
end
data = cat(4,data{:});
score = cat(2,score{:});

idx = randperm(numel(score));
sel = idx(1:floor((numel(score))*0.99));
set = 3*ones(1,numel(score));
set(sel) = 1;
imdb.images.data = data ;
imdb.images.score = score;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
