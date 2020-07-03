measurement=200
while [ $measurement -ge 160 ]
do
    sleep 30s
    gpu_us=`nvidia-smi | grep % | awk '{print $(NF-2)}' | awk -F '%' '{print $1}'`
    gpu_mu=`nvidia-smi | grep % | awk '{print $(NF-6)}' | awk -F 'MiB' '{print $1}'`
    #echo "gpu显存利用率："$gpu_mu
    while [ $gpu_us -gt 50 ] && [ $gpu_mu -gt 4096 ]
    do
	sleep 5m
	#echo "gpu正在使用"
	gpu_us=`nvidia-smi | grep % | awk '{print $(NF-2)}' | awk -F '%' '{print $1}'`
	gpu_mu=`nvidia-smi | grep % | awk '{print $(NF-6)}' | awk -F 'MiB' '{print $1}'`
	echo "gpu利用率："$gpu_us" | 显存占用(MB)："$gpu_mu
    done
    echo "measurements:" $measurement
    python train_lista.py --fault_prob 0.1 --SNR 20 --measurements $measurement --sample_nums 300000 --n_epoch 15 --lr 0.0003 --ft_lr 0.0003 
    measurement=$[measurement - 40]
done
