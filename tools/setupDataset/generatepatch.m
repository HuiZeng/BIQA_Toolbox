function patches = generatepatch(I,patchsize,patchnum,patchstep)

[x1,x2,~] = size(I);
X = 1:patchstep:x1-patchsize;
Y = 1:patchstep:x2-patchsize;
cnt = 1;
for p = 1:length(X)
    for q = 1:length(Y)
        patches(:,:,:,cnt) = I(X(p):X(p)+patchsize-1,Y(q):Y(q)+patchsize-1,:);
        cnt = cnt+1;
    end
end

idx = randperm(cnt-1);
sel = idx(1:min(patchnum,numel(idx)));
patches = patches(:,:,:,sel);