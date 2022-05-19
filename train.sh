python3 train.py --n_epochs 20 \
                  --root ../dataset \
                  --batch_size 32 \
                  --model convnext_base \
                  --optimizer sgd \
                  --lr 7e-4

python3 train.py --n_epochs 20 \
                  --root ../dataset/image_size1024 \
                  --batch_size 8 \
                  --model convnext_small \
                  --optimizer sgd \
                  --lr 7e-4

python3 train.py --n_epochs 20 \
                  --root ../dataset/image_size1024 \
                  --batch_size 16 \
                  --model resnext50 \
                  --lr 7e-4
