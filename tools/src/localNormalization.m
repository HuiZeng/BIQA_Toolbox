function out_img = localNormalization(I,f)


% filter_size = size(f,1);
% if filter_size > 1
%     I = gpuArray(single(I));
%     mu = imfilter(I,f);
%     I2 = I - mu;
%     sgm = sqrt(imfilter(I2.^2,f)) + eps;
%     out_img = gather(I2 ./ sgm);
% %     out_img = gather(I - mu);
% else
%     out_img = im2single(I);
% end
% out_img = out_img(filter_size:end-filter_size+1,filter_size:end-filter_size+1,:);

% % out_img = im2single(I);
% [x,y,~] = size(I);
% scale = 256.0/min(x,y);
% I = imresize(I,scale);        
out_img = I;