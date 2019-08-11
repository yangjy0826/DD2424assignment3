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