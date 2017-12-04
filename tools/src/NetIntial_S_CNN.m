function net = NetIntial_S_CNN(varargin)

opts.batchNormalization = true ;
opts.networkType = 'dagnn' ;
opts.lossType = 'CE';
opts.scale = 1 ;
opts.class = 1 ;
opts.epoch = 10;
opts.weightInitMethod = 'xavierimproved' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

net.layers = {} ;
start_size = 32;
pad = 0;
net.layers{end+1} = struct('type', 'conv','weights', {{init_weight(opts, 3, 3, 3, start_size, 'single'),zeros(1,start_size,'single')}}, 'stride', 1, 'pad', pad) ;
net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [2 2], 'stride', 2, 'pad', pad) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', 'weights', {{init_weight(opts, 3, 3, start_size, start_size*2, 'single'),zeros(1,start_size*2,'single')}}, 'stride', 1, 'pad', pad) ;
net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [2 2], 'stride', 2, 'pad', pad) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', 'weights', {{init_weight(opts, 3, 3, start_size*2, start_size*4, 'single'),zeros(1,start_size*4,'single')}}, 'stride', 1, 'pad', pad) ;  
net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [2 2], 'stride', 2, 'pad', pad) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', 'weights', {{init_weight(opts, 3, 3, start_size*4, start_size*8, 'single'),zeros(1,start_size*8,'single')}}, 'stride', 1, 'pad', pad) ;  
net.layers{end+1} = struct('type', 'pool', 'method', 'max', 'pool', [2 2], 'stride', 2, 'pad', pad) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', 'weights', {{init_weight(opts, 2, 2, start_size*8, start_size * 16, 'single'),zeros(1,start_size * 16,'single')}}, 'stride', 1, 'pad', pad) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout','rate', 0.5) ;
net.layers{end+1} = struct('type', 'conv', 'weights', {{init_weight(opts, 1, 1, start_size * 16, opts.class, 'single'),zeros(1,opts.class,'single')}}, 'stride', 1, 'pad', pad) ;
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 50 * sqrt(opts.class) ;
end
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
net.renameVar('input','data');
net.addLayer('myLoss', dagnn.myLoss('lossType',opts.lossType), {'x16','label'}, 'objective') ;

% optionally switch to batch normalization
if opts.batchNormalization
  net = insertBnorm(net, 1) ;
  net = insertBnorm(net, 4) ;
  net = insertBnorm(net, 7) ;
  net = insertBnorm(net, 10) ;
end

% Meta parameters
net.meta.inputSize = [64 64 3] ;
net.meta.trainOpts.learningRate = logspace(-2, -3, opts.epoch);
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.trainOpts.batchSize = 1024 ;


% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out))/2 ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
