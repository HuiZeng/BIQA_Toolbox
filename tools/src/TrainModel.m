function [net, info] = TrainModel(varargin)


opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'dagnn' ;
opts.PatchSize = 64;
opts.PatchNum = 50;
opts.Model = 'AlexNet';
opts.database = 'ChallengeDB_release';
opts.trainingpropertion = 0.8;
opts.seed = 1;
opts.Net = @NetIntial_S_CNN;
opts.quantizationMethod = 'uniform';
opts.bins = 5;
opts.beta = 10;
opts.PatchSize = 224;
opts.PatchStep = 16;
opts.epoch = 10;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.bins == 1
    opts.lossType = 'MSE';
else
    opts.lossType = 'CE';
end

opts.expDir = fullfile('data',[opts.Model '_' opts.database '_TrainPropertion'...
              num2str(opts.trainingpropertion) '_PatchSize' num2str(opts.PatchSize)...
              '_PatchNum' num2str(opts.PatchNum) '_Seed' num2str(opts.seed),...
              '_bins' num2str(opts.bins), '_beta', num2str(opts.beta)]) ;
opts.dataDir = fullfile('databases', opts.database) ;
opts.imdbPath = fullfile('data', ['imdb_' opts.database '_PatchSize' num2str(opts.PatchSize)...
              '_PatchNum' num2str(opts.PatchNum) '.mat']);
opts.train = struct();
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;



% --------------------------------------------------------------------
%                                 model intialization                        
% --------------------------------------------------------------------

if isempty(opts.network)
    if strcmp(opts.Model,'AlexNet')
      net = load(fullfile('pretrained_models','imagenet-caffe-alex.mat')); 
      net = vl_simplenn_tidy(net);
      net.layers = net.layers(1:end-2);
      net.layers{end+1} = struct('name','dropout1','type', 'dropout','rate', 0.5) ;
      net.layers{end+1} = struct('name','pred','type', 'conv', 'weights',...
                          {{randn(1, 1, 4096, opts.bins, 'single')*0.01,zeros(1,opts.bins,'single')}},...
                          'stride', 1, 'pad', 0) ;
      net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
      net.renameVar('input','data');
      net.addLayer('myLoss', dagnn.myLoss('lossType',opts.lossType), {'x21','label'}, 'objective') ;
      move(net, 'gpu')
      net.meta.inputSize = [opts.PatchSize opts.PatchSize 3] ;
      net.meta.trainOpts.learningRate = logspace(-3, -4, opts.epoch);
      net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
      net.meta.trainOpts.batchSize = 256 ;
      net.meta.trainOpts.numSubBatches = 1 ;
        net.meta.trainOpts.weightDecay = 0.0005;
    elseif strcmp(opts.Model,'ResNet')
        netStruct = load(fullfile('pretrained_models','imagenet-resnet-50-dag.mat')) ;
        net = dagnn.DagNN.loadobj(netStruct) ;
        net.removeLayer('fc1000');
        net.removeLayer('prob');
        dropoutBlock = dagnn.DropOut('rate',0.5);
        net.addLayer('dropout',dropoutBlock,{'pool5'},{'pool5d'},{});
        fc_size = [1 1 2048 opts.bins];
        fc5Block = dagnn.Conv('size',fc_size,'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
        net.addLayer('pred',fc5Block,{'pool5d'},{'pred'},{'fc5_filter','fc5_bias'});   
        f_ind = net.layers(end).paramIndexes(1);
        b_ind = net.layers(end).paramIndexes(2);
        he_gain = sqrt(2/(fc_size(1)*fc_size(2)*fc_size(3)));
        net.params(f_ind).value = (1/2)*he_gain*randn(fc_size, 'single');
        net.params(b_ind).value = zeros(fc_size(4),1, 'single');
        net.addLayer('myLoss', dagnn.myLoss('lossType',opts.lossType), {'pred','label'}, 'objective') ;
        move(net, 'gpu')
        net.meta.inputSize = {'input', [224,224,3,64]} ;
        net.meta.trainOpts.learningRate = logspace(-3, -4, opts.epoch);
        net.meta.trainOpts.numEpochs = opts.epoch;
        net.meta.trainOpts.batchSize = 64 ;
        net.meta.trainOpts.numSubBatches = 1 ;
        net.meta.trainOpts.weightDecay = 0.0001 ;
    elseif strcmp(opts.Model,'S_CNN')
        net = opts.Net('batchNormalization', opts.batchNormalization,...
                       'networkType', opts.networkType, 'lossType', opts.lossType,...
                       'class',opts.bins,'epoch',opts.epoch) ;
    end
end

%% for pre-processing, no use currently
filter_size = 7;
opts.filter_ = gpuArray(fspecial('gaussian',filter_size,1.0));

% --------------------------------------------------------------------
%                                 dataset  preparison                     
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
    switch opts.database
        case 'LIVE'
            imdb = getImdbLIVE(opts) ;
        case 'TID2013'
            imdb = getImdbTID2013(opts) ;
        case 'CSIQ'
            imdb = getImdbCSIQ(opts) ;
        case 'ChallengeDB_release'
            imdb = getImdbChalleng(opts) ;
        otherwise
            error('Unknown database!');
    end

    if opts.bins > 1
        if strcmp(opts.quantizationMethod,'LloydMax')
            [partition,codebook] = lloyds(imdb.images.score,opts.bins);
            imdb.weights = codebook;
            D = bsxfun(@minus,imdb.images.score,imdb.weights');
            D = exp(-opts.beta*(D.^2));
            D = bsxfun(@rdivide,D,sum(D));
            imdb.images.labels = D;
        else 
            partition = 1.0 / opts.bins;
            imdb.weights = partition/2:partition:1.001;
            D = bsxfun(@minus,imdb.images.score,imdb.weights');
            D = exp(-opts.beta*(D.^2));
            D = bsxfun(@rdivide,D,sum(D));
            imdb.images.labels = D;
        end
        scores_train = imdb.images.score(1:opts.PatchNum:end)';
        D = D(:,1:opts.PatchNum:end)';
        svm_model = svmtrain(scores_train, D, '-s 4 -t 0');
        [scores_pred, ~,~] = svmpredict(scores_train, D, svm_model,'-q');
        svm_err = (sum(abs(scores_pred - scores_train))) / length(scores_train);
    else
        imdb.images.labels = imdb.images.score;
    end
end
dataMean = mean(imdb.images.data, 4);
imdb.images.data_mean = dataMean;

try
    net.meta.classes.weights = imdb.weights;
    net.meta.svm_model = svm_model;
    net.meta.svm_err = svm_err;
end
net.meta.normalization.averageImage = dataMean;


% --------------------------------------------------------------------
%                          model training
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train_modified ;
  case 'dagnn', trainfn = @cnn_train_dag_modified ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;


% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
images = bsxfun(@minus, single(images), imdb.images.data_mean) ;
labels = single(imdb.images.labels(:,batch));
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'data', images, 'label', labels} ;
