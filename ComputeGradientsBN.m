function [grad_W, grad_b] = ComputeGradientsBN(X, Y, P, W, lambda, h, s_hat, s, u, v)
n = size(X,2);
[~,k_layer]=size(W);
grad_W=cell(1,k_layer);
grad_b=cell(1,k_layer);
% 1
g = -(Y-P)';

% 2
grad_b{k_layer}=mean(g,1)';
grad_W{k_layer}=1/n*(h{k_layer-1}*g)'+2*lambda*W{k_layer};

% 3
g = g * W{k_layer};
g = g.*(s_hat{k_layer-1}>0)';

% 4
for i=k_layer-1:-1:1
    g = BatchNormBackPass(g, s{i}, u{i}, v{i});
%     mean(g,1)
    grad_b{i}=mean(g,1)';
    if i>1
        grad_W{i}=1/n*(h{i-1}*g)'+2*lambda*W{i};
        g=g * W{i};
        g = g.*(s_hat{i-1}>0)';
    else
        grad_W{i}=1/n*(X*g)'+2*lambda*W{i};
    end
end