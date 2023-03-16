@echo off

set CONFIG=%1
set WORKDIR=%2
set SCENE=%3

echo "python run.py --config %CONFIG%\coarse.py -p %WORKDIR% --no_reload --no_dvgo_init --sdf_mode voxurf_womask_coarse --scene %SCENE%"
python run.py --config %CONFIG%\coarse.py -p %WORKDIR% --no_reload --no_dvgo_init --sdf_mode voxurf_womask_coarse --scene %SCENE%

echo "python run.py --config %CONFIG%\fine.py --render_test -p %WORKDIR% --no_reload --no_dvgo_init --sdf_mode voxurf_womask_fine --scene %SCENE%"
python run.py --config %CONFIG%\fine.py --render_test -p %WORKDIR% --no_reload --no_dvgo_init  --sdf_mode voxurf_womask_fine --scene %SCENE%
