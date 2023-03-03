root=$1
video_path=$2

python process_video.py --source_dir $root --video_path $video_path --mode get_frames

rembg p ${root}"/image" ${root}"/image_rmbg"
echo "Done with rembg"

python process_video.py --source_dir $root --mode get_masks