function [score] = processOneImage_CNN(net,img_path)

I = imread(img_path);
filter_size = 7;
filter_ = gpuArray(fspecial('gaussian',filter_size,1.0));
I = localNormalization(I,filter_);
patchsize = size(net.meta.normalization.averageImage,1);

if strcmp(net.meta.modelType,'DirectTraining')
    step = patchsize / 2;
else
    step = 64;
end
[x1,x2,~] = size(I);
X = 1:step:x1-patchsize;
Y = 1:step:x2-patchsize;
data = zeros(patchsize,patchsize,3,length(X)*length(Y),'single');
cnt = 1;
for p = 1:length(X)
    for q = 1:length(Y)
        data(:,:,:,cnt) = I(X(p):X(p)+patchsize-1,Y(q):Y(q)+patchsize-1,:);
        cnt = cnt+1;
    end
end

data = bsxfun(@minus, single(data), net.meta.normalization.averageImage);
data = gpuArray(data) ;

net.eval({'data', data}) ;
index = find(arrayfun(@(a) strcmp(a.name, 'myLoss'), net.layers)==1);
tlayerName = net.layers(index).inputs{1};
score = net.vars(net.getVarIndex(tlayerName)).value ;
if strcmp(net.layers(end).block.lossType,'CE')
    score = squeeze(gather(vl_nnsoftmax(score)));
else
    score = squeeze(gather(score));
end


if size(score,2) == 1
    score = mean(score);
else
%     score_map = net.meta.classes.weights;
%     score = score_map * score;
    [score, ~,~] = svmpredict([1:size(score,2)]', double(score'), net.meta.svm_model1,'-q');
    score = mean(score);
end

end
