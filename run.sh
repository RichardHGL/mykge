
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode rand --f --n_sample 100 --v 100
#CUDA_VISIBLE_DEVICES=1 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode rela --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode popu --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode adve --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode cach --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode kall --eval --f
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode 1all --eval --f

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode rand --f --n_sample 100 --v 101
#CUDA_VISIBLE_DEVICES=1 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode rela --eval --f
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode popu --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode cach --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 100 --batch_size2 50 --bern --mode kall --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 100 --batch_size2 100 --bern --mode 1all --eval --f --v 1

#CUDA_VISIBLE_DEVICES=7 python -u main.py --model TransH --data YAGO3-10 --dim 64 --batch_size 256 --batch_size2 25 --bern --mode rand --n_sample 500 --f &
#CUDA_VISIBLE_DEVICES=4 python -u main.py --model TransH --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode rela &
#CUDA_VISIBLE_DEVICES=5 python -u main.py --model TransH --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode popu &
#CUDA_VISIBLE_DEVICES=6 python -u main.py --model TransH --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode adve &
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model TransH --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode cach &
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model TransH --data YAGO3-10 --dim 100 --batch_size 50 --batch_size2 25 --bern --mode kall --f
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model TransH --data YAGO3-10 --dim 64 --batch_size 50 --batch_size2 50 --bern --mode 1all --f


#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode rand --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode rela --f --eval
#CUDA_VISIBLE_DEVICES=2 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode popu --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode adve --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode cach --f --eval
#CUDA_VISIBLE_DEVICES=2 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode kall --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode 1all --f --eval

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode rand --f --eval
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode rela --f -eval
#CUDA_VISIBLE_DEVICES=4 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode popu --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode adve --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode cach --f --eval
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 128 --batch_size2 50 --bern --mode kall --f --eval
#CUDA_VISIBLE_DEVICES=1 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 128 --batch_size2 50 --bern --mode 1all --f --eval --v 1

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 25 --bern --mode rand --f --n_sample 100 &
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode rela &
#CUDA_VISIBLE_DEVICES=1 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode popu &
#CUDA_VISIBLE_DEVICES=5 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 50 --bern --mode adve --f --eval
#CUDA_VISIBLE_DEVICES=6 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 512 --batch_size2 5 --bern --mode cach --f --eval
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 100 --batch_size2 50 --bern --mode kall --f
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model DistMult --data YAGO3-10 --dim 64 --batch_size 50 --batch_size2 50 --bern --mode 1all --f


#CUDA_VISIBLE_DEVICES=0 python -u main.py --model ConvE --data FB15K237 --dim 200 --batch_size 128 --batch_size2 128 --bern --mode kall --f --eval --v 1
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model ConvE --data FB15K237 --dim 200 --batch_size 128 --batch_size2 256 --bern --mode 1all --f --eval

#CUDA_VISIBLE_DEVICES=3 python -u main.py --model ConvE --data WN18RR   --dim 200 --batch_size 128 --batch_size2 256 --bern --mode kall --f --eval
#CUDA_VISIBLE_DEVICES=2 python -u main.py --model ConvE --data WN18RR   --dim 200 --batch_size 128 --batch_size2 256 --bern --mode 1all --f --eval

#CUDA_VISIBLE_DEVICES=2 python -u main.py --model ConvE --data YAGO3-10 --dim 200 --batch_size 128 --batch_size2 50 --bern --mode kall --f --eval
#CUDA_VISIBLE_DEVICES=6 python -u main.py --model ConvE --data YAGO3-10 --dim 200 --batch_size 128 --batch_size2 50 --bern --mode 1all --f --eval


#CUDA_VISIBLE_DEVICES=6 python -u main.py --model ConvTransE --data FB15K237 --dim 200 --batch_size 128 --batch_size2 128 --bern --mode kall --f --eval --v 2
#CUDA_VISIBLE_DEVICES=1 python -u main.py --model ConvTransE --data FB15K237 --dim 200 --batch_size 128 --batch_size2 128 --bern --mode 1all --f --eval

#CUDA_VISIBLE_DEVICES=4 python -u main.py --model ConvTransE --data WN18RR   --dim 200 --batch_size 256 --batch_size2 256 --bern --mode kall --f --eval
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model ConvTransE --data WN18RR   --dim 200 --batch_size 256 --batch_size2 256 --bern --mode 1all --f --eval

#CUDA_VISIBLE_DEVICES=5 python -u main.py --model ConvTransE --data YOGO3-10 --dim 200 --batch_size 256 --batch_size2 256 --bern --mode rand 
#CUDA_VISIBLE_DEVICES=4 python -u main.py --model ConvTransE --data YAGO3-10 --dim 200 --batch_size 256 --batch_size2 50 --bern --mode kall --f --eval


#CUDA_VISIBLE_DEVICES=7 python -u main.py --model Kbgan --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --scratch --f --eval
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model Kbgan --data WN18RR   --dim 100 --batch_size 128 --batch_size2 25 --scratch --f --eval
#CUDA_VISIBLE_DEVICES=7 python -u main.py --model Kbgan --data YAGO3-10 --dim 100 --batch_size 128 --batch_size2 25 --scratch &


# Graph
CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode grap --n_sample 12 --v 12.001 --rate 0.01 --f --adv
#CUDA_VISIBLE_DEVICES=1 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode grap --n_sample 100 --rate 0.02 --v 2 &
#CUDA_VISIBLE_DEVICES=2 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode grap --n_sample 100 --rate 0.05 --v 5 &
#CUDA_VISIBLE_DEVICES=3 python -u main.py --model DistMult --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode grap --n_sample 100 --rate 0.10 --v 10 & 

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model ConvE --data FB15K237 --dim 200 --batch_size 128 --batch_size2 128 --bern --mode 1vsN --n_sample 100 --rate 0.1 --v 10 --f
#CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model ConvE --data FB15K237 --dim 200 --batch_size 128 --batch_size2 128 --bern --mode 1vsN --n_sample 100 --rate 0.01 --v 100.01 --f &
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model ConvE --data WN18RR   --dim 200 --batch_size 128 --batch_size2 16 --bern --mode 1vsN --n_sample 200 --rate 0.0 --v 1 --trained --pretrained_name ConvE-WN18RR-1vsN-1 --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model ConvE --data WN18RR   --dim 200 --batch_size 128 --batch_size2 16 --bern --mode 1vsN --n_sample 100 --rate 0.02 --v 100.02 --f

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode grap --n_sample 12  --rate 0.01 --adv --f --v 12.001
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode grap --n_sample 50 --rate 0.01 --adv --f --v 50

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode rand --n_sample 100 --v 100 --trained --pretrained_name TransH-FB15K237-rand-100 --eval --f
#CUDA_VISIBLE_DEVICES=0 python -u main.py --model TransH --data WN18RR   --dim 100 --batch_size 256 --batch_size2 100 --bern --mode cach --n_sample 12 --v 12 --trained --pretrained_name TransH-WN18RR-grap--10.01 --eval --f

#CUDA_VISIBLE_DEVICES=0 python -u main.py --model DistMult --data FB15K237 --dim 100 --batch_size 256 --batch_size2 256 --bern --mode cach --n_sample 12 --v 12 --adv  --f

