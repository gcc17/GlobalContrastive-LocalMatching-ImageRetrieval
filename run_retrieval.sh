for dataset_name in 'oxford5k' 'paris6k'
do

# python perform_retrieval.py \
#     --dataset_file_path /home/chen/cv_proj/retrieval_data/gnd_r${dataset_name}.mat \
#     --images_dir /home/chen/cv_proj/retrieval_data/${dataset_name}_images \
#     --query_global_path output/features_20/${dataset_name}/query_global.npy \
#     --index_global_path output/features_20/${dataset_name}/index_global.npy \
#     --query_local_path output/features_20/${dataset_name}/query_local.pickle \
#     --index_local_path output/features_20/${dataset_name}/index_local.pickle 

# python perform_retrieval.py \
#     --dataset_file_path /home/chen/cv_proj/retrieval_data/gnd_r${dataset_name}.mat \
#     --images_dir /home/chen/cv_proj/retrieval_data/${dataset_name}_images \
#     --query_global_path output/features_200/${dataset_name}/query_global.npy \
#     --index_global_path output/features_200/${dataset_name}/index_global.npy \
#     --query_local_path output/features_200/${dataset_name}/query_local.pickle \
#     --index_local_path output/features_200/${dataset_name}/index_local.pickle

python perform_retrieval.py \
    --dataset_file_path /home/chen/cv_proj/retrieval_data/gnd_r${dataset_name}.mat \
    --images_dir /home/chen/cv_proj/retrieval_data/${dataset_name}_images \
    --query_global_path output/supcon/features_200_001/${dataset_name}/query_global.npy \
    --index_global_path output/supcon/features_200_001/${dataset_name}/index_global.npy \
    --query_local_path output/supcon/features_200_001/${dataset_name}/query_local.pickle \
    --index_local_path output/supcon/features_200_001/${dataset_name}/index_local.pickle  

done