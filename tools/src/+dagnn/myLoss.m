classdef myLoss < dagnn.Loss
    
  properties
    lossType = 'CE'
  end
  
  methods
    function outputs = forward(obj, inputs, params)
        X = inputs{1};
        c = inputs{2};
        c = reshape(c,size(X));
        switch obj.lossType
            case 'CE'
                X = vl_nnsoftmax(X);
                Y = squeeze(- c .* log(X./(c  + eps(1))));
            case 'MSE'
                Y = squeeze((X - c).^2);                                    
        end
        outputs{1} = sum(Y(:));
        n = obj.numAveraged ;
        m = n + size(inputs{1},4);
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        X = inputs{1};
        c = inputs{2};
        c = reshape(c,size(X));
        switch obj.lossType
            case 'CE'
                X = vl_nnsoftmax(X);
                Y = X - c;
            case 'MSE'
                Y = X - c;
        end
        derInputs = {Y, []};
        derParams = {};  
    end

    function obj = myLoss(varargin)
      obj.load(varargin) ;
    end
  end
end


