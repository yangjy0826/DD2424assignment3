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
% % %cost function
% % lambda = 0; % when lambda = 0, there is no regularization
% % J = ComputeCostBN(Xtrain, Ytrain, W, b, lambda);
% % %accuracy
% % acc = ComputeAccuracyBN(Xtrain, ytrain, W, b);
% % %gradients                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
% % [ngrad_b, ngrad_W] = ComputeGradsNumSlowBN(Xtrain, Ytrain, W, b, lambda, 1e-5);
% % [P, s, s_hat, h, u, v] = EvaluateClassifierBN(Xtrain, W, b); %evaluate
% % [grad_W, grad_b] = ComputeGradientsBN(Xtrain, Ytrain, P, W, lambda, h, s_hat, s, u, v);
% % %gradient check
% % eps=0.001;
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
% % J = ComputeCostBN(Xtrain(:, 1:n), Ytrain(:, 1:n), W, b, lambda);
% J = ComputeCostBN(Xtrain, Ytrain, W, b, lambda);
% %accuracy
% acc = ComputeAccuracyBN(Xtrain, ytrain, W, b);
% %gradients                                                                       
% [P, s, s_hat, h, u, v] = EvaluateClassifierBN(Xtrain, W, b); %evaluate
% [grad_W, grad_b] = ComputeGradientsBN(Xtrain, Ytrain, P, W, lambda, h, s_hat, s, u, v);
% eps=0.001;
% 
% n_batch = 100; %the number of images in a mini-batch
% n_epochs = 10; %the number of runs through the whole training set
% rho=0.9; %momentum parameter:{0.5,0.9,0.99}
% eta=0.3;
% GDparams = [n_batch, eta, n_epochs, rho];
% % [Wstar_t, bstar_t, J2] = MiniBatchGDmoBN(Xtrain, Ytrain,GDparams, W, b, lambda);%with momentum
% % plot(J2);
% % grid on;
% % xlabel('epoch');
% % ylabel('loss');
% % legend('with batch normalization');
% 
% [Wstar_t, bstar_t, loss_t, loss_v] = MiniBatchGDmoBN2(Xtrain, Ytrain, Xvalid, Yvalid, GDparams, W, b, lambda);
% figure();
% plot(loss_t);
% hold on;
% plot(loss_v);
% grid on;
% title(['eta= ',num2str(eta),', with batch normalization']);
% legend('training loss','validation loss');
% xlabel('epoch');
% ylabel('loss');
% hold off;
% 
% %accuracy
% accuracy_train = ComputeAccuracyBN(Xtrain, ytrain, Wstar_t, bstar_t);
% acc_valid = ComputeAccuracyBN(Xvalid, yvalid, Wstar_t, bstar_t);
% accuracy_test = ComputeAccuracyBN(Xtest, ytest, Wstar_t, bstar_t);

%% Coarse to fine search
% mini-batch
n_batch = 100; %the number of images in a mini-batch
n_epochs = 50; %the number of runs through the whole training set
rho=0.9; %momentum parameter:{0.5,0.9,0.99}
l_min=-6;
l_max=-1;%3;
e_min=-2.5;%-6;
e_max=-0.5;%-1;
for i=1:100
l(i) = l_min + (l_max - l_min)*rand(1, 1);
lambda(i) = 10^l(i);%the learning rate
e(i) = e_min + (e_max - e_min)*rand(1, 1);
eta(i) = 10^e(i);%the learning rate
GDparams = [n_batch, eta(i), n_epochs, rho];
[Wstar_t, bstar_t, J] = MiniBatchGDmo(Xtrain, Ytrain,GDparams, W, b, lambda(i));
%accuracy
% accuracy_train = ComputeAccuracy(Xtrain, ytrain, Wstar_t, bstar_t);
acc_valid(i) = ComputeAccuracy(Xvalid, yvalid, Wstar_t, bstar_t);
% accuracy_test = ComputeAccuracy(Xtest, ytest, Wstar_t, bstar_t);
end

%Draw loss picture
figure();
scatter3(l,e,acc_valid);
hold off;
grid on;
xlabel('lambda');
ylabel('eta');
zlabel('accuracy');