function y = processOneImage(refImagePath,disImgPath,is_no_reference,is_color,algorithm_name)

    if ~is_no_reference
        refI = imread(refImagePath);                
    end
    I = imread(disImgPath);
    if is_color
        if is_no_reference
            y = feval(algorithm_name, I);
        else
            y = feval(algorithm_name, refI, I);
        end
    else
        grayI = rgb2gray(I);
        if is_no_reference
            y = feval(algorithm_name, grayI);
        else
            grayrefI = rgb2gray(refI);
            y = feval(algorithm_name, grayrefI, grayI);
        end
    end
end