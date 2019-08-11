function g = BatchNormBackPass(g, s, u, v)
n = size(s,2);
grad_v=sum(0.5*g.*power(v+eps,-3/2)'.*(s-u)');
grad_u=-sum(g.*power(v+eps,-1/2)');
g=g.*power(v+eps,-1/2)' + 2/n*grad_v.*(s-u)'+grad_u/n;
end