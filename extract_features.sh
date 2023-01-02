for dataset_name in 'oxford5k' 'paris6k'
do

for extract_feature_set in 'local' 'global'
do

for image_set in 'query' 'index'
do
    
    python extract_features.py \
        --extract_feature_set $extract_feature_set \
        --weight_path output/checkpoints/model_epoch_0200.pyth \
        --dataset_file_path /home/chen/cv_proj/retrieval_data/gnd_r${dataset_name}.mat \
        --images_dir /home/chen/cv_proj/retrieval_data/${dataset_name}_images \
        --image_set $image_set \
        --output_features_dir output/features_200/${dataset_name}/

done

done

done