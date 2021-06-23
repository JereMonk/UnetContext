import sys
import yaml
from getData import get_generator
#from build_Unet import build_Unet
from build_Unet import Unet
#from build_Wnet import Wnet
#from ncut_loss import compute_soft_ncuts
import numpy as np
import tensorflow as tf
from trainer import training 
import os

def main(arg):

    EXP_FOLDER= arg[0]

    print(arg[0])

    with open(EXP_FOLDER+"/custom.yaml", 'r') as stream:
        custom_data = yaml.safe_load(stream)

    INPUT_DIM = int(custom_data['INPUT']['DIM'])
    
    #TRAIN_DATASET = custom_data["TRAIN_DATASET"]
    #TEST_DATASET = custom_data["TEST_DATASET"]

    JSON_PATHS_TRAIN = custom_data["JSON_PATHS_TRAIN"]
    JSON_PATHS_TEST = custom_data["JSON_PATHS_TEST"]

    #SUBCATS = custom_data["SUBCATS"]
    #K = int(custom_data["MODEL"]["K"])
    #STAGES = (np.arange(1,int(custom_data["MODEL"]["STAGES"])+1))
    #FILTERS = int(custom_data["MODEL"]["FILTERS"])
    MAX_ITER = int(custom_data["SOLVER"]["MAX_ITER"])
    IMS_PER_BATCH = int(custom_data["SOLVER"]["IMS_PER_BATCH"])
    BASE_LR = float(custom_data["SOLVER"]["BASE_LR"])
    STEPS = [int(custom_data["SOLVER"]["STEPS"][0]),int(custom_data["SOLVER"]["STEPS"][1])]
    
    CHECKPOINT_PERIOD = int(custom_data["CHECKPOINT_PERIOD"])
    IMG_PERIOD =int( custom_data["IMG_PERIOD"])
    TEST_PERIOD = int(custom_data["TEST_PERIOD"])

    USE_DROPOUT = bool(int(custom_data["USE_DROPOUT"]))

    DECAY_STEP = int(custom_data["SOLVER"]["DECAY_STEP"])
    DECAY_RATE= int(custom_data["SOLVER"]["DECAY_RATE"])

    SIGMA = int(custom_data['INPUT']['SIGMA'])
    BLUR_KERNEL = int(custom_data['INPUT']['BLUR_KERNEL'])
    NOISE_AMP = int(custom_data['INPUT']['NOISE_AMP'])

    L1_REG = int(custom_data['L1_REG'])
    L2_REG = int(custom_data['L2_REG'])

    RECONSTRUCTION_LOSS_WEIGHT = int(custom_data["RECONSTRUCTION_LOSS_WEIGHT"])
    print('RECONSTRUCTION_LOSS_WEIGHT',RECONSTRUCTION_LOSS_WEIGHT)

    
    N_HOLES = int(custom_data['N_HOLES'])
    MAX_HEIGHT = int(custom_data['MAX_HEIGHT'])
    MAX_WIDTH= int(custom_data['MAX_WIDTH'])
    MIN_WIDTH = int(custom_data['MIN_WIDTH'])
    MIN_HEIGHT = int(custom_data['MIN_HEIGHT'])
    FILL_VALUE = (custom_data['FILL_VALUE'])

  
    
    
    ## DATA
    generator_train = get_generator(json_paths=JSON_PATHS_TRAIN,size=INPUT_DIM,batch_size=IMS_PER_BATCH,damaged=False,
    n_holes=N_HOLES,max_height=MAX_HEIGHT,max_width=MAX_WIDTH,min_width = MIN_WIDTH,min_height = MIN_HEIGHT,fill_value_mode = FILL_VALUE)
    generator_test = get_generator(json_paths=JSON_PATHS_TEST,size=INPUT_DIM,batch_size=IMS_PER_BATCH,damaged=False,
    n_holes=N_HOLES,max_height=MAX_HEIGHT,max_width=MAX_WIDTH,min_width = MIN_WIDTH,min_height = MIN_HEIGHT,fill_value_mode = FILL_VALUE)
    

    #encoder = build_Unet(K=K,stages = STAGES,filters = FILTERS,type='encoder',input_size=INPUT_DIM,use_dropout=USE_DROPOUT)
    #decoder = build_Unet(K=K,stages = STAGES,filters = FILTERS,type='decoder',input_size=INPUT_DIM,use_dropout=USE_DROPOUT)
    
    encoder = Unet(K=3,type='decoder',input_size=INPUT_DIM,do_dropout=USE_DROPOUT,l1_reg=L1_REG,l2_reg=L2_REG)
    #decoder = Unet(K=K,type='decoder',input_size=INPUT_DIM,do_dropout=USE_DROPOUT,l1_reg=L1_REG,l2_reg=L2_REG)
    #wn = Wnet(encoder,decoder,(INPUT_DIM,INPUT_DIM))

    ## Load weights
    start_iter=0
    ckpts=[]

    if('checkpoint' in os.listdir(EXP_FOLDER)):
        with open(EXP_FOLDER+"/checkpoint", 'r') as stream:
            try:
                ckpts = yaml.safe_load(stream)
                last_ckpt = ckpts["model_checkpoint_path"]
                encoder.load_weights(EXP_FOLDER+'/'+last_ckpt)
                start_iter = int(last_ckpt.replace('ckpt',''))

            except yaml.YAMLError as exc:
                print(exc)
                  
    print('Starting from iter : ', start_iter)

    

    # Compile the model
    encoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
        loss = tf.keras.losses.MeanSquaredError()
    )

    training(model=encoder,train_dataset=generator_train,test_dataset=generator_test,max_iter=MAX_ITER,start_iter=start_iter,base_lr=BASE_LR,ckpt_freq=CHECKPOINT_PERIOD,img_freq=IMG_PERIOD,dir_path=EXP_FOLDER,solver_steps=STEPS,test_freq=TEST_PERIOD,reconstruction_loss_weight=RECONSTRUCTION_LOSS_WEIGHT,decay_step=DECAY_STEP,decay_rate=DECAY_RATE,sigma=SIGMA,blur_kernel=BLUR_KERNEL,noise_amp=NOISE_AMP)
   
  

if __name__ == "__main__":
    
    main(sys.argv[1:])
