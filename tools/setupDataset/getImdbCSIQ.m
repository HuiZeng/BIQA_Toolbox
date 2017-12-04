% --------------------------------------------------------------------
function imdb = getImdbCSIQ(opts)
% --------------------------------------------------------------------

mos = load(fullfile('databases', 'CSIQ', 'dmos.txt'));
mos = (mos - min(mos)) / (max(mos) - min(mos));
disClasses = dir(fullfile('databases', 'CSIQ','dst_imgs'));
ref_list = dir(fullfile('databases', 'CSIQ','src_imgs', '*.png'));

count = 0;
for i=1:length(disClasses)
    if strcmp(disClasses(i,1).name,'.') || strcmp(disClasses(i,1).name,'..')
        continue;
    end
    currentDisClassDirName = fullfile('CSIQ','dst_imgs', disClasses(i,1).name);
    files = dir(fullfile('databases',currentDisClassDirName, '*.png'));

    for j=1:length(files)
        idx = strfind(files(j,1).name,'.');
        refImg = [files(j,1).name(1:idx(1)) 'png'];
        count = count + 1;
        imageList{count} = fullfile('databases',currentDisClassDirName, files(j,1).name);
        for p = 1:numel(ref_list)
            if strcmp(refImg,ref_list(p).name) == 1
                classes(count) = p;
                break;
            end
        end
    end
end 

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
imdb.images.data = data ;
imdb.images.score = score;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:opts.classes,'uniformoutput',false) ;