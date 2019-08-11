function J = ComputeCostBN(X, Y, W, b, lambda)
[P, ~, ~, ~, ~, ~] = EvaluateClassifierBN(X, W, b);
% l = -log(Y'*P);
% l=diag(l);
% J = sum(sum(l))/size(X,2) + lambda*sum(sum(W{1}.^2))+ lambda*sum(sum(W{2}.^2));
l = -log(sum(Y.*P,1));
J = sum(l)/size(X,2);
for i=1:size(W, 2) % add the regularization term
    J = J + lambda*sum(sum(W{i}.^2));
end
end