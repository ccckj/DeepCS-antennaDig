function AoAgen(sample_size)
theta = unifrnd(0,pi/2,1,sample_size);
phi = unifrnd(0,pi/2,1,sample_size);
save('theta.mat','theta')
save('phi.mat','phi')
end