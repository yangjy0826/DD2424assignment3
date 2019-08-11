clc;
clear;
addpath Datasets/cifar-10-matlab/cifar-10-batches-mat/;

% Read in the data & initialize the parameters

%Use part of the data
[Xtrain,Ytrain,ytrain] = LoadBatch('data_batch_1.mat'); % training data
% Xmean = mean(Xtrain,2);
% Xtrain = Xtrain - Xmean;
[Xvalid,Yvalid,yvalid] = LoadBatch('data_batch_2.mat'); % validation data
[Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data
% Xvalid = Xvalid - Xmean;
% Xtest = Xtest - Xmean;

%%% Use all data
% [Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat'); % training data part1
% [Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat'); % training data part2
% [Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat'); % training data part3
% [Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat'); % training data part4
% [X5,Y5,y5] = LoadBatch('data_batch_5.mat'); % training data part5
% 
% Xtrain5=X5(:,1:size(X5,2)-1000);
% Xtrain=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
% Ytrain5=Y5(:,1:size(Y5,2)-1000);
% Ytrain=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
% ytrain5=y5(:,1:size(X5,2)-1000);
% ytrain=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];
% 
% Xvalid=X5(:,(size(X5,2)-999):size(X5,2));
% Yvalid=Y5(:,(size(Y5,2)-999):size(Y5,2));
% yvalid=y5(:,(size(y5,2)-999):size(y5,2));
% 
% [Xtest,Ytest,ytest] = LoadBatch('test_batch.mat'); % test data
%%% Use all data end

mean_X = mean(Xtrain, 2);
Xtrain = Xtrain - repmat(mean_X, [1, size(Xtrain, 2)]);
Xvalid = Xvalid - repmat(mean_X, [1, size(Xvalid, 2)]);
Xtest = Xtest - repmat(mean_X, [1, size(Xtest, 2)]);

% %Use small amount of data during gradient check, becausse gradient check is very time consuming
% d=100;%dimention
% n=100;
% Xtrain=Xtrain(1:d,1:n);
% ytrain=ytrain(1:n);
% Ytrain=Ytrain(:,1:n);
% Xvalid=Xvalid(1:d,1:n);
% yvalid=yvalid(1:n);
% Yvalid=Yvalid(:,1:n);
% Xtest=Xtest(1:d,1:n);
% ytest=ytest(1:n);
% Ytest=Ytest(:,1:n);

k=2; %number of layers
% m={50,30}; % no. of hidden units in each hidden layer, the size of this cell should be  k-1
m={50};
[W,b]=initialize(Xtrain,k,m); 
%% Without batch norm
% % % Gradient check
% % %cost function
% % lambda = 0; % when lambda = 0, there is no regularization
% % % J = ComputeCost(Xtrain(:, 1:n), Ytrain(:, 1:n), W, b, lambda);
% % J = ComputeCost(Xtrain, Ytrain, W, b, lambda);
% % %accuracy
% % acc = ComputeAccuracy(Xtrain, ytrain, W, b);
% % %gradients
% % h = 1e-5;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
% % [ngrad_b, ngrad_W] = ComputeGradsNumSlow(Xtrain, Ytrain, W, b, lambda, h);
% % P = EvaluateClassifier(Xtrain, W, b); %evaluate
% % [grad_W, grad_b] = ComputeGradients(Xtrain, Ytrain, P, W,b, lambda);
% % %gradient check
% % dif_b1=abs(ngrad_b{1}-grad_b{1})./max(eps,sum(abs(ngrad_b{1})+ abs(ngrad_b{1})));
% % re_b1=max(max(dif_b1));
% % dif_b2=abs(ngrad_b{2}-grad_b{2})./max(eps,sum(abs(ngrad_b{2})+ abs(ngrad_b{2})));
% % re_b2=max(max(dif_b2));
% % dif_b3=abs(ngrad_b{3}-grad_b{3})./max(eps,sum(abs(ngrad_b{3})+ abs(ngrad_b{3})));
% % re_b3=max(max(dif_b3));
% % % dif_b4=abs(ngrad_b{4}-grad_b{4})./max(eps,sum(abs(ngrad_b{4})+ abs(ngrad_b{4})));
% % % re_b4=max(max(dif_b4));
% % dif_W1=abs(ngrad_W{1}-grad_W{1})./max(eps,sum(abs(ngrad_W{1})+ abs(ngrad_W{1})));
% % re_W1=max(max(dif_W1));
% % dif_W2=abs(ngrad_W{2}-grad_W{2})./max(eps,sum(abs(ngrad_W{2})+ abs(ngrad_W{2})));
% % re_W2=max(max(dif_W2));
% % dif_W3=abs(ngrad_W{3}-grad_W{3})./max(eps,sum(abs(ngrad_W{3})+ abs(ngrad_W{3})));
% % re_W3=max(max(dif_W3));
% % % dif_W4=abs(ngrad_W{4}-grad_W{4})./max(eps,sum(abs(ngrad_W{4})+ abs(ngrad_W{4})));
% % % re_W4=max(max(dif_W4));
% 
% %train
% lambda = 0.001; % when lambda = 0, there is no regularization
% % J = ComputeCost(Xtrain(:, 1:n), Ytrain(:, 1:n), W, b, lambda);
% % J = ComputeCost(Xtrain, Ytrain, W, b, lambda);
% %accuracy
% % acc = ComputeAccuracy(Xtrain, ytrain, W, b);
% %gradients
% h = 1e-5;                                                                        
% % P = EvaluateClassifier(Xtrain, W, b); %evaluate
% % [grad_W, grad_b] = ComputeGradients(Xtrain, Ytrain, P, W,b, lambda);
% 
% %mini-batch
% n_batch = 100; %the number of images in a mini-batch
% n_epochs = 10; %the number of runs through the whole training set
% rho=0.9; %momentum parameter:{0.5,0.9,0.99}
% %lambda=0;
% eta=0.3;
% GDparams = [n_batch, eta, n_epochs, rho];
% % [Wstar_t, bstar_t, J1] = MiniBatchGDmo(Xtrain, Ytrain,GDparams, W, b, lambda);%with momentum
% % plot(J1);
% % grid on;
% % xlabel('epoch');
% % ylabel('loss');
% % legend('without batch normalization');
% 
% [Wstar_t, bstar_t, loss_t, loss_v] = MiniBatchGDmo2(Xtrain, Ytrain, Xvalid, Yvalid, GDparams, W, b, lambda);
% figure();
% plot(loss_t);
% hold on;
% plot(loss_v);
% grid on;
% title(['eta= ',num2str(eta),', without batch normalization']);
% legend('training loss','validation loss');
% xlabel('epoch');
% ylabel('loss');
% hold off;
% 
% %accuracy
% accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
% acc_valid = ComputeAccuracy(Xvalid, yvalid, Wstar_t, bstar_t);
% accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);

%% With batch norm
% %cost function
% lambda = 0; % when lambda = 0, there is no regularization
% J = ComputeCostBN(Xtrain, Ytrain, W, b, lambda);
% %accuracy
% acc = ComputeAccuracyBN(Xtrain, ytrain, W, b);
% %gradients                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
% [ngrad_b, ngrad_W] = ComputeGradsNumSlowBN(Xtrain, Ytrain, W, b, lambda, 1e-5);
% [P, s, s_hat, h, u, v] = EvaluateClassifierBN(Xtrain, W, b); %evaluate
% [grad_W, grad_b] = ComputeGradientsBN(Xtrain, Ytrain, P, W, lambda, h, s_hat, s, u, v);
% %gradient check
% eps=0.001;
% dif_b1=abs(ngrad_b{1}-grad_b{1})./max(eps,sum(abs(ngrad_b{1})+ abs(ngrad_b{1})));
% re_b1=max(max(dif_b1));
% dif_b2=abs(ngrad_b{2}-grad_b{2})./max(eps,sum(abs(ngrad_b{2})+ abs(ngrad_b{2})));
% re_b2=max(max(dif_b2));
% dif_b3=abs(ngrad_b{3}-grad_b{3})./max(eps,sum(abs(ngrad_b{3})+ abs(ngrad_b{3})));
% re_b3=max(max(dif_b3));
% % dif_b4=abs(ngrad_b{4}-grad_b{4})./max(eps,sum(abs(ngrad_b{4})+ abs(ngrad_b{4})));
% % re_b4=max(max(dif_b4));
% dif_W1=abs(ngrad_W{1}-grad_W{1})./max(eps,sum(abs(ngrad_W{1})+ abs(ngrad_W{1})));
% re_W1=max(max(dif_W1));
% dif_W2=abs(ngrad_W{2}-grad_W{2})./max(eps,sum(abs(ngrad_W{2})+ abs(ngrad_W{2})));
% re_W2=max(max(dif_W2));
% dif_W3=abs(ngrad_W{3}-grad_W{3})./max(eps,sum(abs(ngrad_W{3})+ abs(ngrad_W{3})));
% re_W3=max(max(dif_W3));
% % dif_W4=abs(ngrad_W{4}-grad_W{4})./max(eps,sum(abs(ngrad_W{4})+ abs(ngrad_W{4})));
% % re_W4=max(max(dif_W4));

%train
lambda = 0.001; % when lambda = 0, there is no regularization
% J = ComputeCostBN(Xtrain(:, 1:n), Ytrain(:, 1:n), W, b, lambda);
J = ComputeCostBN(Xtrain, Ytrain, W, b, lambda);
%accuracy
acc = ComputeAccuracyBN(Xtrain, ytrain, W, b);
%gradients                                                                       
[P, s, s_hat, h, u, v] = EvaluateClassifierBN(Xtrain, W, b); %evaluate
[grad_W, grad_b] = ComputeGradientsBN(Xtrain, Ytrain, P, W, lambda, h, s_hat, s, u, v);
eps=0.001;

n_batch = 100; %the number of images in a mini-batch
n_epochs = 10; %the number of runs through the whole training set
rho=0.9; %momentum parameter:{0.5,0.9,0.99}
eta=0.3;
GDparams = [n_batch, eta, n_epochs, rho];
% [Wstar_t, bstar_t, J2] = MiniBatchGDmoBN(Xtrain, Ytrain,GDparams, W, b, lambda);%with momentum
% plot(J2);
% grid on;
% xlabel('epoch');
% ylabel('loss');
% legend('with batch normalization');

[Wstar_t, bstar_t, loss_t, loss_v] = MiniBatchGDmoBN2(Xtrain, Ytrain, Xvalid, Yvalid, GDparams, W, b, lambda);
figure();
plot(loss_t);
hold on;
plot(loss_v);
grid on;
title(['eta= ',num2str(eta),', with batch normalization']);
legend('training loss','validation loss');
xlabel('epoch');
ylabel('loss');
hold off;

%accuracy
accuracy_train = ComputeAccuracyBN(Xtrain, ytrain, Wstar_t, bstar_t);
acc_valid = ComputeAccuracyBN(Xvalid, yvalid, Wstar_t, bstar_t);
accuracy_test = ComputeAccuracyBN(Xtest, ytest, Wstar_t, bstar_t);

%% Coarse to fine search
% % mini-batch
% n_batch = 100; %the number of images in a mini-batch
% n_epochs = 50; %the number of runs through the whole training set
% rho=0.9; %momentum parameter:{0.5,0.9,0.99}
% l_min=-6;
% l_max=-1;%3;
% e_min=-2.5;%-6;
% e_max=-0.5;%-1;
% for i=1:100
% l(i) = l_min + (l_max - l_min)*rand(1, 1);
% lambda(i) = 10^l(i);%the learning rate
% e(i) = e_min + (e_max - e_min)*rand(1, 1);
% eta(i) = 10^e(i);%the learning rate
% GDparams = [n_batch, eta(i), n_epochs, rho];
% [Wstar_t, bstar_t, J] = MiniBatchGDmo(Xtrain, Ytrain,GDparams, W, b, lambda(i));
% %accuracy
% % accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
% acc_valid(i) = ComputeAccuracy(Xvalid, yvalid, Wstar_t, bstar_t);
% % accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);
% end
% 
% %Draw loss picture
% figure();
% scatter3(l,e,acc_valid);
% hold off;
% grid on;
% xlabel('lambda');
% ylabel('eta');
% zlabel('accuracy');

%% functions
function [X, Y, y] = LoadBatch(filename)
 A = load(filename);
 X = double(A.data)/double(255); %normalized to figures between 0 and 1
 %X is of type "double"
 y = A.labels;
 [a,~] = size(y);
 K = 10;
 Y = zeros(a,K);
 for i = 1:a
 Y(i,y(i)+1) = 1;  % y after one-hot encoding
 end
 X = X';
 Y = Y';
 y = y';
end

function [W,b]=initialize(X,k_layer,m)
K = 10; % size of the output layer
d = size(X,1); %dimention, all is 3072
% rng(200);
% sigma=0.01;
n=[d,m,K];
W=cell(1,k_layer); % no. of layers of the network
b=cell(1,k_layer);

%%%%%%%%%%%%%%%%%%%%%%
mean = 0;
sigma = 0.001;

for i=1:k_layer-1
W{i} = mean + sigma*randn(m{i},d);
d = m{i};
b{i} = zeros(m{i},1);
end
W{k_layer} = mean + sigma*randn(K,m{end});
b{k_layer} = zeros(K,1);
b{k_layer} = mean + sigma*rand(K,1);
%%%%%%%%%%%%%%%%%%%%%%%

% for i=1:k_layer
%     sum_nodes = sum(cell2mat(m))+ K;%the sum of nodes in all layers, in our case 50+30+10
%     sigma = sqrt(2/sum_nodes); %He initialization, to initilize the weights
%     W{i} = sigma*randn([n{i+1} n{i}]); 
%     % In our case, the size of W1 is m1*d,
%     % the size of W2 is m2*m1, the size of W3 is K*m2
%     b{i} = sigma*randn([n{i+1} 1]);
%     % In our case, the size of W1 is m1*1,
%     % the size of W2 is m2*1, the size of W3 is K*1
% end
end

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

function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
% For two layers network
% l = -log(Y'*P);
% l=diag(l);
% J = sum(sum(l))/size(X,2) + lambda*sum(sum(W{1}.^2)) + lambda*sum(sum(W{2}.^2));
l = -log(sum(Y.*P,1));
J = sum(l)/size(X,2); %divided by the number of training data
for i=1:size(W, 2) % add the regularization term
    J = J + lambda*sum(sum(W{i}.^2));
end
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
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

function [Wstar, bstar, J] = MiniBatchGDmo(X, Y,GDparams, W, b, lambda)
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
    GDparams(2)=GDparams(2)*0.95; 
%     if mod(i, 10) == 0 %This is add for bonus point 1
%         GDparams(2)=GDparams(2)*0.1; %This is add for bonus point 1
%     end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end

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

function [Wstar, bstar, J] = MiniBatchGDmoBN(X, Y,GDparams, W, b, lambda)
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

        %P = EvaluateClassifier(Xbatch, W, b);
        [P, s, s_hat, h, u, v]= EvaluateClassifierBN(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradientsBN(Xbatch, Ybatch, P, W, lambda, h, s_hat, s, u, v);
        %with momentum 
        for k=1:k_layer
            v_W{k}=GDparams(4)*v_W{k}+GDparams(2)*grad_W{k};
            W{k} = W{k} - v_W{k};
            v_b{k}=GDparams(4)*v_b{k}+GDparams(2)*grad_b{k};
            b{k}= b{k}- v_b{k};
        end
    end
%     J(i) = ComputeCost(X, Y, W, b, lambda);
    J(i) = ComputeCostBN(X, Y, W, b, lambda);
    GDparams(2)=GDparams(2)*0.95; 
%     if mod(i, 10) == 0 %This is add for bonus point 1
%         GDparams(2)=GDparams(2)*0.1; %This is add for bonus point 1
%     end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end

function [Wstar, bstar, J, J2] = MiniBatchGDmoBN2(X, Y, X2, Y2, GDparams, W, b, lambda)
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

        %P = EvaluateClassifier(Xbatch, W, b);
        [P, s, s_hat, h, u, v]= EvaluateClassifierBN(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradientsBN(Xbatch, Ybatch, P, W, lambda, h, s_hat, s, u, v);
        %with momentum 
        for k=1:k_layer
            v_W{k}=GDparams(4)*v_W{k}+GDparams(2)*grad_W{k};
            W{k} = W{k} - v_W{k};
            v_b{k}=GDparams(4)*v_b{k}+GDparams(2)*grad_b{k};
            b{k}= b{k}- v_b{k};
        end
    end
%     J(i) = ComputeCost(X, Y, W, b, lambda);
    J(i) = ComputeCostBN(X, Y, W, b, lambda);
    J2(i) = ComputeCostBN(X2, Y2, W, b, lambda);
    GDparams(2)=GDparams(2)*0.95; 
%     if mod(i, 10) == 0 %This is add for bonus point 1
%         GDparams(2)=GDparams(2)*0.1; %This is add for bonus point 1
%     end %This is add for bonus point 1
end
Wstar = W;
bstar = b;
end

function g = BatchNormBackPass(g, s, u, v)
n = size(s,2);
grad_v=sum(0.5*g.*power(v+eps,-3/2)'.*(s-u)');
grad_u=-sum(g.*power(v+eps,-1/2)');
g=g.*power(v+eps,-1/2)' + 2/n*grad_v.*(s-u)'+grad_u/n;
end