project_dir=/data5/store1/dlt/rectified_flow/
export PYTHONPATH=$project_dir:$PYTHONPATH
main_path=${project_dir}/main/darcy.py
cfg_path=${project_dir}/main/config_darcy.py
python $main_path $cfg_path