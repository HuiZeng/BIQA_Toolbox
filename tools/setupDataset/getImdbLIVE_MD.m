% --------------------------------------------------------------------
function imdb = getImdbLIVE_MD(opts)
% --------------------------------------------------------------------

List1 = load(fullfile('databases', opts.database, 'Part 1', 'Imagelists.mat'));
Scores1 = load(fullfile('databases', opts.database, 'Part 1', 'Scores.mat'));
refImgList1 = fullfile('databases', opts.database, 'Part 1','blurjpeg', List1.refimgs(List1.ref4dist));
DisImgList1 = fullfile('databases', opts.database, 'Part 1','blurjpeg', List1.distimgs);

List2 = load(fullfile('databases', opts.database, 'Part 2', 'Imagelists.mat'));
Scores2 = load(fullfile('databases', opts.database, 'Part 2', 'Scores.mat'));
refImgList2 = fullfile('databases', opts.database, 'Part 2','blurnoise', List2.refimgs(List2.ref4dist));
DisImgList2 = fullfile('databases', opts.database, 'Part 2','blurnoise', List2.distimgs);

RefImgList = cat(1,refImgList1,refImgList2);
DisImgList = cat(1,DisImgList1,DisImgList2);
dmos = [Scores1.DMOSscores, Scores2.DMOSscores];
classes = [List1.ref4dist; List1.ref4dist]';
dmos = (dmos - min(dmos)) / (max(dmos) - min(dmos));

[trainingSet testingSet] = generateTrainingSet(classes,2.^(opts.seed));

for i = 1:numel(trainingSet)
    fprintf('Finished image %d / %d...\n', i, numel(trainingSet));
    I = imread(DisImgList{trainingSet(i)});
    I = localNormalization(I,opts.filter_);
    data{i} = generatepatch(I,opts.PatchSize,opts.PatchNum,opts.PatchStep);
    score{i} = ones(1,size(data{i},4)) * dmos(trainingSet(i));
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