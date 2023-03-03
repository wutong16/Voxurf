dir=$1

echo "Making mask ..."
python process_video.py --source_dir $dir --mode get_masks --rmbg_img_folder rgb --white_bg

echo "Camera converting ..."
python convert_cameras.py --source_dir $dir --mode tat02neus

echo "Camera normalizing ..."
python preprocess_cameras.py --source_dir $dir