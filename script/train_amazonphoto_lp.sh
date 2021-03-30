gpu=0
seed=1234
save=0

export HGCAE_HOME=$(pwd)
export DATAPATH="$HGCAE_HOME/data"

n_head=1
lr=0.01
normalize_feats=0
att_logit=exp

model=HGCAE
att_type=sparse_adjmask_dist
dataset=amazon-photo   
act=relu
c=1.0
c_trainable=0
dropout=0
weight_decay=0.001
hidden_dim=32
dim=32
lambda_rec=0.001
manifold=PoincareBall
optimizer=Adam

python train_solver.py --model $model \
    --seed $seed --dataset $dataset --lr $lr --normalize-feats $normalize_feats\
    --min-epoch 100 --save $save --log-freq 10 --cuda $gpu\
    --hidden-dim $hidden_dim --dim $dim --num-layers 2 --act $act --bias 1 \
    --dropout $dropout --weight-decay $weight_decay \
    --alpha 0.2 --n-heads $n_head \
    --manifold $manifold --c $c --c-trainable $c_trainable \
    --lambda-rec $lambda_rec \
    --optimizer $optimizer\
    --use-att 1 --att-type $att_type --att-logit $att_logit 
