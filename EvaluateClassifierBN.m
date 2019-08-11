function [P, s, s_hat, h, u, v]= EvaluateClassifierBN(X, W, b)
n = size(X,2);
[~,k_layer]=size(W);
s=cell(1,k_layer);
h=cell(1,k_layer);
u=cell(1,k_layer); %mean
v=cell(1,k_layer); %variance
% V=cell(1,k_layer); %variance
u_rep=cell(1,k_layer); %mean
v_rep=cell(1,k_layer); %mean
s_hat=cell(1,k_layer);
for i=1:k_layer-1
    b{i} = repmat(b{i},1,n);
    if i==1
        s{i} = W{i}*X+b{i};
    else
        s{i} = W{i}*h{i-1}+b{i};
    end
    m=size(s{i},1);
    u{i}=1/n*sum(s{i},2); %mean
    v{i}=zeros(m,1);
    u_rep{i} = repmat(u{i},1,n);
    v{i}= var(s{i},0,2)*(size(s{i},2)-1)/size(s{i},2);
%     for j=1:m
%         u_rep{i} = repmat(u{i},1,n);
%         v{i}(j)=1/m*sum((s{i}(j,:)-u_rep{i}(j)).^2);
%     end
%     V{i}=diag(v{i}+eps);
    v_rep{i} = repmat(v{i},1,n);
    s_hat{i}=(s{i}-u_rep{i})./sqrt(v_rep{i}+eps);
    s_vec=cell2mat(s_hat(1,i));
    h{i} = max(0, s_vec);
end
s{k_layer} = W{k_layer}*h{k_layer-1}+b{k_layer};
s_vec=cell2mat(s(1,k_layer));
P = softmax(s_vec);
end
