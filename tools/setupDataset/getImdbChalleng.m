% --------------------------------------------------------------------
function imdb = getImdbChalleng(opts)
% --------------------------------------------------------------------

opts.dataDir = fullfile('databases', 'ChallengeDB_release');
imageList = load(fullfile(opts.dataDir,'Data','AllImages_release.mat'));
mos = load(fullfile(opts.dataDir,'Data','AllMOS_release.mat'));
mos_std = load(fullfile(opts.dataDir,'Data','AllStdDev_release.mat'));
imageList = fullfile(opts.dataDir,'Images',imageList.AllImages_release(8:end));
mos = mos.AllMOS_release(8:end);
mos_std = mos_std.AllStdDev_release(8:end);
mos = (mos - min(mos)) / (max(mos) - min(mos));
trainingNum = ceil(length(mos)*opts.trainingpropertion);
rng(opts.seed-1);
randomsplit = randperm(length(mos),length(mos));
trainingSet = randomsplit(1:trainingNum);

for i = 1:numel(trainingSet)
    fprintf('Finished image %d / %d...\n', i, numel(trainingSet));
    I = imread(imageList{trainingSet(i)});
    I = localNormalization(I,opts.filter_);
    data{i} = generatepatch(I,opts.PatchSize,opts.PatchNum,opts.PatchStep);
    score{i} = ones(1,size(data{i},4)) * mos(trainingSet(i));
    score_std{i} = ones(1,size(data{i},4)) * mos_std(trainingSet(i));
end
data = cat(4,data{:});
score = cat(2,score{:});
score_std = cat(2,score_std{:});

idx = randperm(numel(score));
sel = idx(1:floor((numel(score))*0.99));
set = 3*ones(1,numel(score));
set(sel) = 1;
imdb.images.data = data ;
imdb.images.score = score;
imdb.images.score_std = score_std;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
