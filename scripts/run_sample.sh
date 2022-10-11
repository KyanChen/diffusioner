
FLAGS="--batch_size 16 --model_name V2 --split test --timestep_respacing 50 --use_ddim False"

out_dir=./out_dir/levir_trainval_V2_e60/result_e60_step50_2
model_path=./out_dir/levir_trainval_V2_e60/model600000.pt
python ./scripts/cd_res_sample.py --out_dir=$out_dir $FLAGS --model_path $model_path

out_dir=./out_dir/levir_trainval_V2_e60/result_e60_step50_3
python ./scripts/cd_res_sample.py --out_dir=$out_dir $FLAGS --model_path $model_path
out_dir=./out_dir/levir_trainval_V2_e60/result_e60_step50_4
python ./scripts/cd_res_sample.py --out_dir=$out_dir $FLAGS --model_path $model_path
out_dir=./out_dir/levir_trainval_V2_e60/result_e60_step50_5
python ./scripts/cd_res_sample.py --out_dir=$out_dir $FLAGS --model_path $model_path
