.PHONY: start train eval 

# 변수 설정 (필요에 따라 경로 및 파라미터 수정)
PYTHON = python

model_path = ./res/DDFF12_ep700_b20_full_diff1_from_DFF-DFV/best.tar
my_model_path = ./res/deformable_1000_b20_best/best.tar


# 실행 명령어
train:
	$(PYTHON) train.py --dataset DDFF12 --DDFF12_pth ./data/DFF/my_ddff_trainVal.h5 --epochs 1500 --savemodel ./res/ --stack_num 5
    
train_new:
	$(PYTHON) train_new.py --dataset DDFF12 --DDFF12_pth ./data/DFF/my_ddff_trainVal.h5 --epochs 700 --savemodel ./res/ --stack_num 5 --use_diff 1

eval:
	$(PYTHON) eval_DDFF12.py --stack_num 5 --loadmodel $(model_path)
    
eval_new:
	$(PYTHON) eval.py --stack_num 5 --loadmodel $(model_path)