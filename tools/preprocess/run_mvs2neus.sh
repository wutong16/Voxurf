dir=$1

 echo "Removing background ..."
 rembg p ${dir}"/images" ${dir}"/images_rmbg"
 python process_video.py --source_dir $dir --mode get_masks

 echo "Camera converting ..."
 python convert_cameras.py --source_dir $dir --mode mvs2neus

 echo "Camera normalizing ..."
python preprocess_cameras.py --source_dir $dir