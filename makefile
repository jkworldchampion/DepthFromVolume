.PHONY: start train eval 

# 변수 설정 (필요에 따라 경로 및 파라미터 수정)
PYTHON = python
# model_path = ./res/Attention_1000_diff/best.tar
model_path = ./res/DFF-DFV.tar


# 실행 명령어
train:
	$(PYTHON) train.py --dataset DDFF12 --DDFF12_pth ./data/DFF/my_ddff_trainVal.h5 --epochs 1500 --savemodel ./res/ --stack_num 5
    
train_val:
	$(PYTHON) train_val.py --dataset DDFF12 --DDFF12_pth ./data/DFF/my_ddff_trainVal.h5 --epochs 1000 --savemodel ./res/ --stack_num 5

eval:
	$(PYTHON) eval_DDFF12.py --stack_num 5 --loadmodel $(model_path)
    
standard_eval:
	$(PYTHON) eval.py --stack_num 5 --loadmodel $(model_path)