script -c "python3 main.py --dataset vaihingen_1000 --network resnet50 --exp_name vaihingen_plus250 --epochs 500 --batch_size 3 --num_workers 3 --n_classes 6 --lr 1e-5 --new_data_size 250 --data_augmentation False" logs/vaihingen_plus250.txt

python3 test.py --dataset vaihingen_1000 --network resnet50 --exp_name vaihingen_plus250 --model_path models/vaihingen_plus250/vaihingen_plus250_best.pth --n_classes 6 --save_imgs True
