function [trainingSet, testingSet] = generateTrainingSet(classes,seed,Trainingproportion)

trainingSet = [];
testingSet = [];
N = max(classes);
rng(seed-1);
randomsplit = randperm(N,N);
trainNum = ceil(N * Trainingproportion);
trainingID = randomsplit(1:trainNum);
testID = randomsplit(trainNum+1:end);
for i = 1:numel(trainingID)
    sel = find(classes == trainingID(i));
    trainingSet = [trainingSet sel];
end
for i = 1:numel(testID)
    sel = find(classes == testID(i));
    testingSet = [testingSet sel];
end
 