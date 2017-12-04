function [score] = BLIINDS2(I)

features = bliinds2_feature_extraction(I);
score = bliinds_prediction(features(:)');

end