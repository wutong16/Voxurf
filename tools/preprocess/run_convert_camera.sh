root=$1

python colmap_poses/pose_utils.py --source_dir $root
python convert_cameras.py --source_dir $root
python preprocess_cameras.py --source_dir $root