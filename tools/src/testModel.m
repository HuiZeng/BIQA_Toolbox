function [SRCC,PLCC] = testModel(ModelType, net, testDatabase,seed,Propertion)

net.meta.modelType = ModelType;
testMethod = 'CNN';
switch testDatabase
    case 'LIVE'
        [SRCC,PLCC] = test_LIVE(net,testDatabase,testMethod, Propertion, seed);
    case 'TID2013'
        [SRCC,PLCC] = test_TID2013(net,testDatabase,testMethod, Propertion, seed);
    case 'CSIQ'
        [SRCC,PLCC] = test_CSIQ(net,testDatabase,testMethod, Propertion, seed);
    case 'ChallengeDB_release'
        [SRCC,PLCC] = test_ChallengeDB(net,testDatabase,testMethod, Propertion, seed);
    otherwise
        error('Unknown database!');
end
