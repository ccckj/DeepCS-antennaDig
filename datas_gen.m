%%%this file aims to generate training samples
clc
clear
%--------------------------parameter configuration-------------------------
antenna_x = 16;
antenna_y = 16;
antenna_size = antenna_x * antenna_y;
sample_size = antenna_size;   % times of mearsurment for one sample
probability = 0.1; %  probability of antenna fault
sample_nums  = 3e5;     %numbers of sample for training
snr = 10;  %db
smooth_nums = 10;     %smooth the noise
%-------------------------------end----------------------------------------
rho = 10^(snr./10);
w = ones(antenna_size,1);  % antenna weight matriax

%theta = unifrnd(0,pi/2,1,sample_size);
%phi = unifrnd(0,pi/2,1,sample_size);
try
    load('theta.mat')
    load('phi.mat')
catch
    AoAgen(sample_size);
    load('theta.mat')
    load('phi.mat')
end


datas = zeros(sample_nums,sample_size);
labels = zeros(sample_nums,antenna_size);
for sn = 1:sample_nums
    if mod(sn,500)==0
        disp([num2str(sn),'_th sample generating...']);
    end
    % set fault factor
    ff = ones(antenna_size,1);
    cf = zeros(antenna_size,1);
    
    
    for i=1: antenna_size
        prob = rand(1);
        if prob<=probability    %%todo
            amp = rand(1);    %todo 0-0.5
            phase = rand(1)*pi/2;  %todo
            ff(i) = amp*exp(1j*phase);
            cf(i) = 1 - ff(i);
        end
    end
    m = 0: antenna_x - 1;
    n = 0: antenna_y - 1;
    y = zeros(sample_size,1);
    for i=1:sample_size
        ax = exp(1j*m*pi*sin(theta(i))*cos(phi(i)));
        ay = exp(1j*n*pi*sin(theta(i))*sin(phi(i)));
        a = kron(ax.',ay.');
        
        % generate noise and smooth
        noise = sqrt(2)./2 * (randn(antenna_size,smooth_nums) + 1j*randn(antenna_size,smooth_nums))./sqrt(rho);
        noise_fault = sqrt(2)./2 * (randn(antenna_size,smooth_nums) + 1j*randn(antenna_size,smooth_nums))./sqrt(rho);
        noise = mean(noise,2);
        noise_fault = mean(noise_fault,2);
        
        noise_res = w.' * noise - (ff .* w).' * noise_fault;
        y(i) = a.' * (cf .* w);
        y(i) = y(i) + noise_res;

        
    end
    datas(sn,:) = y.';
    labels(sn,:) = ff.';
    
end
%todo save cf,ff or cf.* w


data_name = ['./data/datas_n',num2str(sample_nums),'_s',num2str(antenna_size),...
    '_p',num2str(probability*100),'_snr',num2str(snr),'_smo',num2str(smooth_nums),'.mat']
save(data_name,'datas')
label_name = ['./data/labels_n',num2str(sample_nums),'_s',num2str(antenna_size),...
    '_p',num2str(probability*100),'_snr',num2str(snr),'_smo',num2str(smooth_nums),'.mat']
save(label_name, 'labels')
