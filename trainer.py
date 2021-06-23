
import tensorflow as tf
from PIL import Image 
import numpy as np
import PIL 


def rescale(image):
    return( np.array(((image+1)/2)*255 ).astype("uint8") )

def set_learning_rate(step_counter,model,base_lr,steps,decay_step,decay_rate):
    
    if(step_counter<=decay_step):
        new_lr = base_lr
    
    else:
        new_lr = base_lr**(decay_rate*(step_counter//decay_step))
    model.optimizer.lr = new_lr
    
def training(model,train_dataset,test_dataset,max_iter,start_iter,base_lr,ckpt_freq,img_freq,dir_path,solver_steps,test_freq,reconstruction_loss_weight,decay_step,decay_rate,sigma,blur_kernel,noise_amp):
  
    ##TRAIN
    total_train_loss = []


    step_counter=start_iter
    writer = tf.summary.create_file_writer(dir_path)
    
 
    while(step_counter<max_iter):
        


        print("\nStart of iter %d" % (step_counter,))
        print("Learning rate" +str(model.optimizer.lr))


        

        for _, x_batch_train in enumerate(train_dataset):
            step_counter+=1
            set_learning_rate(step_counter,model,base_lr,solver_steps,decay_step=decay_step,decay_rate=decay_rate)

            train_loss = model.train_step(x_batch_train,sigma,blur_kernel,noise_amp)

     

            total_train_loss.append(train_loss.numpy())


            if step_counter%ckpt_freq ==0:
                model.save_weights(dir_path+"/ckpt"+str(step_counter))
            
            if step_counter%1==0:
                final_train_loss= np.mean(np.array(total_train_loss))
            
                
                print("step "+str(step_counter) ) 
                print("TRAIN LOSS : ", final_train_loss)
  
                total_train_loss = []
    

                with writer.as_default():
                    tf.summary.scalar('training lossd', final_train_loss, step=step_counter)
                    tf.summary.scalar('learning rate',model.optimizer.lr , step=step_counter)

          
