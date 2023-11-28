CUDA_VISIBLE_DEVICES=5 \
python instruction_filter.py --model_name_or_path ../output/new-model --instruct_data ./test.json  --instruct_filtered ./result.json
