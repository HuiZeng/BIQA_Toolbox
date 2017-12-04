function feat = CallingTheSourceCodeForFeatureExtraction(imagePath,testMethod)

I = imread(imagePath);

switch testMethod
    case 'BRISQUE'
        feat = brisque_feature(double(rgb2gray(I)));
    case 'DIIVINE'
        feat = divine_feature_extract(rgb2gray(I));
    case 'CORNIA'
        I = rgb2gray(I);
        load('CSIQ_codebook_BS7.mat','codebook0');
        load('LIVE_soft_svm_model.mat','soft_model','soft_scale_param');
        load('CSIQ_whitening_param.mat','M','P');
        feat = CORNIA_Fv(I, codebook0, 'soft', M, P, sqrt(size(codebook0,1)), 10000);
    case 'FRIQUEE'
        feat = extractFRIQUEEFeatures(I);
        feat = feat.friqueeALL;
    case 'HOSA'
        load('whitening_param.mat', 'M', 'P');
        load('codebook_hosa', 'codebook_hosa');
        BS = 7; % patch size
        power = 0.2; % signed power normalizaiton param
        feat = hosa_feature_extraction(codebook_hosa.centroid_cb, codebook_hosa.variance_cb, codebook_hosa.skewness_cb, M, P, BS, power, I);
    otherwise
        error('Unsupported Method!');
end

if size(feat,1) == 1
    feat = feat';
end

end