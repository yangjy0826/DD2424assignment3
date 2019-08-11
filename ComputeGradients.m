function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda)
g = -(Y-P)';
n = size(X,2);
[~,k_layer]=size(W);
s=cell(1,k_layer);
h=cell(1,k_layer);
for i=1:k_layer
    b{i} = repmat(b{i},1,n);
    if i==1
        s{i} = W{i}*X+b{i};
    else
        s{i} = W{i}*h{i-1}+b{i};
    end
    s_vec=cell2mat(s(1,i));
    h{i} = max(0, s_vec);
end
grad_W=cell(1,k_layer);
grad_b=cell(1,k_layer);
for i=k_layer:-1:1
    grad_b{i}=(sum(g,1)/n)';
    if i==1
        grad_W{i} = g'*X'/n+2*lambda*W{i};
    else
        grad_W{i} = g'*h{i-1}'/n+2*lambda*W{i};
        s_vector=cell2mat(s(1,i-1));
        g=g*W{i};
        g=g.*(s_vector>0)'; %ReLu activation
    end
end    
end