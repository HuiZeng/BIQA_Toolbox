% --------------------------------------------------------------------
function imdb = getImdbTID2013(opts)
% --------------------------------------------------------------------

Path = fullfile('databases', 'TID2013');
fp = fopen(fullfile(Path,'mos_with_names.txt'),'r');
for i = 1:3000
    temp = fgetl(fp);
    mos(i) = str2num(temp(1:7));
    classes(i) = str2num(temp(10:11));
    imageList{i} = fullfile(Path,'distorted_images',temp(9:end));
end
fclose(fp);
mos = (mos - min(mos)) / (max(mos) - min(mos));
[trainingSet testingSet] = generateTrainingSet(classes,opts.seed,opts.trainingpropertion);

for i = 1:numel(trainingSet)
    fprintf('Finished image %d / %d...\n', i, numel(trainingSet));
    I = imread(imageList{trainingSet(i)});
    I = localNormalization(I,opts.filter_);
    data{i} = generatepatch(I,opts.PatchSize,opts.PatchNum,opts.PatchStep);
    score{i} = ones(1,size(data{i},4)) * mos(trainingSet(i));
end
data = cat(4,data{:});
score = cat(2,score{:});

idx = randperm(numel(score));
sel = idx(1:floor((numel(score))*0.99));
set = 3*ones(1,numel(score));
set(sel) = 1;
imdb.images.data = data;
imdb.images.score = score;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;