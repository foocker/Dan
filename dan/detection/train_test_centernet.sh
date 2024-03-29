# train
config_p=/root/Codes/Dan/dan/detection/config/res18_centernet.py
model=/aidata/dataset/tianchi/result/exp/ctdet/coco_dla/model_best.pth
# python main.py ctdet --exp_id coco_dla  --arch res_18 --batch_size 16 --master_batch 4 --lr 1.25e-4  --gpus 0,1
# python train_centernet.py ctdet $config_p --exp_id coco_dla  --arch resdcn_18 --batch_size 16 --master_batch 4  --num_epochs 300 --lr 1.25e-4  --gpus 0,1
# python main.py --exp_id coco_dla --batch_size 16 --master_batch 4 --lr 1.25e-4  --gpus 0,1
# test
# python test_centernet.py ctdet $config_p --exp_id coco_dla --not_prefetch_test --load_model $model
python train_centernet.py --test ctdet $config_p --exp_id coco_dla --not_prefetch_test --load_model $model
# python test.py --exp_id coco_dla --not_prefetch_test ctdet --load_model /aidata/dataset/tianchi/result/exp/ctdet/coco_dla/model_best.pth
#demo
# python demo.py ctdet --debug 1 --demo ~/Codes/CenterNet/data/cig_box/images/train/JPEGImages --load_model ~/