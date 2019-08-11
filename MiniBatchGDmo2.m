function [Wstar, bstar, J, J2] = MiniBatchGDmo2(X, Y,X2,Y2,GDparams, W, b, lambda)
N = size(X,2);
[~,k_layer]=size(W);
v_W=cell(1,k_layer);
v_b=cell(1,k_layer);
for i=1:k_layer
    v_W{i}=0;
    v_b{i}=0;
end
for i=1:GDparams(3)
    for j=1:N/GDparams(1)
        j_start = (j-1)*GDparams(1) + 1;
        j_end = j*GDparams(1);
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda);
        %with momentum 
        for k=1:k_layer
            v_W{k}=GDparams(4)*v_W{k}+GDparams(2)*grad_W{k};
            W{k} = W{k} - v_W{k};
            v_b{k}=GDparams(4)*v_b{k}+GDparams(2)*grad_b{k};
            b{k}= b{k}- v_b{k};
        end
    end
    J(i) = ComputeCost(X, Y, W, b, lambda);
    J2(i) = ComputeCost(X2, Y2, W, b, lambda);
    GDparams(2)=GDparams(2)*0.95; 
%     if mod(i, 10) == 0 %This is add for bonus point 1
%         GDparams(2)=GDparams(2)*0.1; %This is add for bonus point 1
%     end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end