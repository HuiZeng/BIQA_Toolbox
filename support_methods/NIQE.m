function [score] = NIQE(I)

 load modelparameters.mat
 
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

score = computequality(I,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, mu_prisparam,cov_prisparam);

end