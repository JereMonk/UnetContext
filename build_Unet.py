import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

class Unet(tf.keras.Model):
    def __init__(self,input_size,type,K,do_dropout=False,l1_reg=0,l2_reg=0):
        super(Unet, self).__init__()
        
        self.do_dropout=do_dropout
        self.K=K
        self.type=type
        self.input_size=input_size
       
        if (type=='encoder'):
          self.inputs = tf.keras.Input(shape=(input_size,input_size,3))
        else:
          self.inputs = tf.keras.Input(shape=(input_size,input_size,K))


        self.conv11 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV11',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN11 = tf.keras.layers.BatchNormalization(name='BN11')
        self.ReLU11 = tf.keras.layers.ReLU()

        self.conv12 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV12',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN12 = tf.keras.layers.BatchNormalization(name='BN12')
        self.ReLU12 = tf.keras.layers.ReLU()
        
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',name='MAXPOOL1')
 
        self.conv21 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV21',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN21 = tf.keras.layers.BatchNormalization(name='BN21')
        self.ReLU21 = tf.keras.layers.ReLU()

        self.conv22 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV22',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN22 = tf.keras.layers.BatchNormalization(name='BN22')
        self.ReLU22= tf.keras.layers.ReLU()

        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',name='MAXPOOL2')

        self.conv31 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV31',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN31 = tf.keras.layers.BatchNormalization(name='BN31')
        self.ReLU31 = tf.keras.layers.ReLU()

        self.conv32 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV32',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN32 = tf.keras.layers.BatchNormalization(name='BN32')
        self.ReLU32 = tf.keras.layers.ReLU()
        
        self.maxpool3= tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',name='MAXPOOL3')

        self.conv41 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV41',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN41 = tf.keras.layers.BatchNormalization(name='BN41')
        self.ReLU41 = tf.keras.layers.ReLU()

        self.conv42 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV42',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN42 = tf.keras.layers.BatchNormalization(name='BN42')
        self.ReLU42 = tf.keras.layers.ReLU()

        self.maxpool4= tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same',name='MAXPOOL3')

        self.conv421 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV41',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN421 = tf.keras.layers.BatchNormalization(name='BN41')
        self.ReLU421 = tf.keras.layers.ReLU()

        self.conv422 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV42',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN422 = tf.keras.layers.BatchNormalization(name='BN42')
        self.ReLU422 = tf.keras.layers.ReLU()


        self.upsample1 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(2,2),strides=(2,2),padding='valid',name='UPSAMPLE1',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        
        self.conv51 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV51',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN51 = tf.keras.layers.BatchNormalization(name='BN51')
        self.ReLU51 = tf.keras.layers.ReLU()

        self.conv52 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV52',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN52 = tf.keras.layers.BatchNormalization(name='BN52')
        self.ReLU52 = tf.keras.layers.ReLU()

        self.upsample2 = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=(2,2),strides=(2,2),padding='valid',name='UPSAMPLE2',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))

        self.conv61 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV61',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN61 = tf.keras.layers.BatchNormalization(name='BN61')
        self.ReLU61 = tf.keras.layers.ReLU()

        self.conv62 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV62',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN62 = tf.keras.layers.BatchNormalization(name='BN62')
        self.ReLU62 =tf.keras.layers.ReLU()

        self.upsample3 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(2,2),strides=(2,2),padding='valid',name='UPSAMPLE3')

        self.conv71 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV71',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN71 = tf.keras.layers.BatchNormalization(name='BN71')
        self.ReLU71 = tf.keras.layers.ReLU()

        self.conv72 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV72',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN72 = tf.keras.layers.BatchNormalization(name='BN72')
        self.ReLU72 = tf.keras.layers.ReLU()

        self.upsample4 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(2,2),strides=(2,2),padding='valid',name='UPSAMPLE3')

        self.conv81 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV71',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN81 = tf.keras.layers.BatchNormalization(name='BN71')
        self.ReLU81 = tf.keras.layers.ReLU()

        self.conv82 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='CONV72',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN82 = tf.keras.layers.BatchNormalization(name='BN72')
        self.ReLU82 = tf.keras.layers.ReLU()

        if(type == 'encoder'):
          self.last_conv = tf.keras.layers.Conv2D(filters=K,kernel_size=(1,1),strides=(1,1),padding='same')
          self.last_acti = tf.keras.layers.Softmax(axis=-1)
        else:
          self.last_conv = tf.keras.layers.Conv2D(filters=3,kernel_size=(1,1),strides=(1,1),padding='same')
          self.last_acti = tf.keras.layers.Activation('tanh')
        
    
    def call(self,x,training=False):
      
      #print(self.inputs)
      #x1=self.inputs(x)

      x1= self.conv11(x)
      x1= self.BN11(x1)
      x1=self.ReLU11(x1)

      x1= self.conv12(x1)
      x1= self.BN12(x1)
      x1=self.ReLU12(x1)
      
      
      x2=self.maxpool1(x1)
      if ( training==True or self.do_dropout==True):
        x2=tf.nn.dropout(x2,0.5)


      x2= self.conv21(x2)
      x2= self.BN21(x2)
      x2=self.ReLU21(x2)

      x2= self.conv22(x2)
      x2= self.BN22(x2)
      x2=self.ReLU22(x2)

      

      x3=self.maxpool2(x2)
      if ( training==True or self.do_dropout==True):
        x3=tf.nn.dropout(x3,0.5)

  

      x3= self.conv31(x3)
      x3= self.BN31(x3)
      x3=self.ReLU31(x3)

      x3= self.conv32(x3)
      x3= self.BN32(x3)
      x3=self.ReLU32(x3)


      x4=self.maxpool3(x3)
      if ( training==True or self.do_dropout==True):
        x4=tf.nn.dropout(x4,0.5)

      x4= self.conv41(x4)
      x4= self.BN41(x4)
      x4=self.ReLU41(x4)

      x4= self.conv42(x4)
      x4= self.BN42(x4)
      x4=self.ReLU42(x4)

      x42=self.maxpool4(x4)
      if ( training==True or self.do_dropout==True):
        x42=tf.nn.dropout(x42,0.5)

      x42= self.conv421(x42)
      x42= self.BN421(x42)
      x42=self.ReLU421(x42)

      x42= self.conv422(x42)
      x42= self.BN422(x42)
      x42=self.ReLU422(x42)
      

      upsample1 = self.upsample1(x42)
      
      x5 = tf.keras.layers.Concatenate(axis=3)([x4, upsample1])
      if ( training==True or self.do_dropout==True):
        x5=tf.nn.dropout(x5,0.5)


      x5= self.conv51(x5)
      x5= self.BN51(x5)
      x5=self.ReLU51(x5)

      x5= self.conv52(x5)
      x5= self.BN52(x5)
      x5=self.ReLU52(x5)


      upsample2 = self.upsample2(x5)
      
      x6 = tf.keras.layers.Concatenate(axis=3)([x3, upsample2])
      if ( training==True or self.do_dropout==True):
        x6=tf.nn.dropout(x6,0.5)


      x6= self.conv61(x6)
      x6= self.BN61(x6)
      x6=self.ReLU61(x6)

      x6= self.conv62(x6)
      x6= self.BN62(x6)
      x6=self.ReLU62(x6)

      upsample3 = self.upsample3(x6)
     
      x7 = tf.keras.layers.Concatenate(axis=3)([x2, upsample3])
      if ( training==True or self.do_dropout==True):
        x7=tf.nn.dropout(x7,0.5)

      x7= self.conv71(x7)
      x7= self.BN71(x7)
      x7=self.ReLU71(x7)

      x7= self.conv72(x7)
      x7= self.BN72(x7)
      x7=self.ReLU72(x7)

      upsample4 = self.upsample4(x7)
     
      x8 = tf.keras.layers.Concatenate(axis=3)([x1, upsample4])
      if ( training==True or self.do_dropout==True):
        x8=tf.nn.dropout(x8,0.5)

      x8= self.conv81(x8)
      x8= self.BN81(x8)
      x8=self.ReLU81(x8)

      x8= self.conv82(x8)
      x8= self.BN82(x8)
      x8=self.ReLU82(x8)

      output = self.last_conv(x8)
      output = self.last_acti(output)

      
      return(output)

    def compile(
        self,
        optimizer,
        loss,
       
    ):
        super(Unet, self).compile()

        self.optimizer = optimizer
        self.loss = loss
       
    @tf.function
    def train_step(self, batch_data,sigma=0.00001,blur_kernel=(10,10),noise_amp=0):
        
        img,img_cutout = batch_data
        

        
        with tf.GradientTape() as tape:
          result_encoder = self.call(img_cutout,training=True)
          loss = self.loss(img,result_encoder)
          # REGULARISATION
          for layer in self.layers:
            loss+=tf.math.reduce_sum(layer.losses)
    
        grads = tape.gradient(loss, self.trainable_variables)
          
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )


      

        return loss

  