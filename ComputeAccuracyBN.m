function acc = ComputeAccuracyBN(X, y, W, b)
[P, ~, ~, ~, ~, ~] = EvaluateClassifierBN(X, W, b);
[~,n] = size(y);
correct = 0;
for i = 1:n
    [~,k(i)] = max(P(:,i));
    if y(i)+1 == k(i)
          correct = correct+1;
    end
end
acc = correct/n;
end