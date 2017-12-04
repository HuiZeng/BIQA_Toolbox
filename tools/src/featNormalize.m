function [trainX, testX, minimums, ranges] = featNormalize(trainX, testX)
    
	minimums = min(trainX, [], 1);
	ranges = max(trainX, [], 1) - minimums + eps;
% 
	trainX = (trainX - repmat(minimums, size(trainX, 1), 1)) ./ repmat(ranges, size(trainX, 1), 1);

	testX = (testX - repmat(minimums, size(testX, 1), 1)) ./ repmat(ranges, size(testX, 1), 1);
    
%     trainX = bsxfun(@rdivide,trainX,sqrt(sum(trainX.^2,2)));
% 	testX = bsxfun(@rdivide,testX,sqrt(sum(testX.^2,2)));
end
