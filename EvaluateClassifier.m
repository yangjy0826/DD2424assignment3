function P = EvaluateClassifier(X, W, b)
% n = size(X,2);
% [~,k_layer]=size(W);
% s=cell(1,k_layer);
% h=cell(1,k_layer);
% for i=1:k_layer
%     b{i} = repmat(b{i},1,n);
%     if i==1
%         s{i} = W{i}*X+b{i};
%     else
%         s{i} = W{i}*h{i-1}+b{i};
%     end
%     s_vec=cell2mat(s(1,i));
%     h{i} = max(0, s_vec);  %activate function: Relu
% end
% P = softmax(s_vec);
% end
n = size(X,2);
[~,k_layer]=size(W);
s=cell(1,k_layer);
h=cell(1,k_layer);
for i=1:k_layer-1
    b{i} = repmat(b{i},1,n);
    if i==1
        s{i} = W{i}*X+b{i};
    else
        s{i} = W{i}*h{i-1}+b{i};
    end
    s_vec=cell2mat(s(1,i));
    h{i} = max(0, s_vec);  %activate function: Relu
end
s{k_layer} = W{k_layer}*h{k_layer-1}+b{k_layer};
s_vec=cell2mat(s(1,k_layer));
P = softmax(s_vec);
end