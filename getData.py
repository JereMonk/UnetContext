import numpy as np
from monk import BBox
import tensorflow as tf
from monk import Dataset
import json
import PIL
from monk.utils.s3.s3path import S3Path

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,json_paths, batch_size=10, dim=(128,128), n_channels=3,shuffle=True,damaged=False,
    n_holes=1,max_height=100,max_width=100,min_width = 50,min_height = 50,fill_value_mode = "zero"
    ):
        
        
        
        self.shuffle = shuffle 
        self.dim = dim 
        self.batch_size = batch_size  
        self.n_channels = n_channels
        self.damaged=damaged
        
        self.n_holes = n_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_width = min_width 
        self.min_height = min_height 
        self.fill_value_mode = fill_value_mode  
        
        jsons_data=[]
        
        for json_path in json_paths:
            with open(json_path) as f:
                json_data = json.load(f)
            jsons_data.append(json_data)
            
        self.filter_json(jsons_data,damaged)

        
        
    def filter_json(self,jsons_data,damaged):
        
        
        filtered_json =[]

        for json_data in jsons_data :
        
            for i in range(0,len(json_data)):
                
                
                if json_data[i]["repair_action"]=='not_damaged' and damaged==False :
                    filtered_json.append(json_data[i])
                elif json_data[i]["repair_action"]!='not_damaged' and json_data[i]["repair_action"]!=None and damaged==True and (json_data[i]["label"]=='scratch' or json_data[i]["label"]=='dent' ) :
                    filtered_json.append(json_data[i])
               
        
        self.filtered_json=filtered_json
        self.list_IDs = np.arange(len(filtered_json)) 
        self.indexes = np.arange(len(filtered_json))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        

    def cutout(self, img ):
       
        h = img.shape[0]
        w = img.shape[1]

        if self.fill_value_mode == 'zero':
            f = np.zeros
            param = {'shape': (h, w, 3)}
        elif self.fill_value_mode == 'one':
            f = np.one
            param = {'shape': (h, w, 3)}
        else:
            f = np.random.uniform
            param = {'low': 0, 'high': 255, 'size': (h, w, 3)}

        mask = np.ones((h, w, 3), dtype=np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            h_l = np.random.randint(self.min_height, self.max_height + 1)
            w_l = np.random.randint(self.min_width, self.max_width + 1)

            y1 = np.clip(y - h_l // 2, 0, h)
            y2 = np.clip(y + h_l // 2, 0, h)
            x1 = np.clip(x - w_l // 2, 0, w)
            x2 = np.clip(x + w_l // 2, 0, w)

            mask[y1:y2, x1:x2, :] = 0

        img = np.where(mask, img, f(**param))
       
        return np.float32(img)


    def load_image(self,id):
        
        data = self.filtered_json[id]
        
        
        if('s3:/monk-client-images/' in data["path"]):
            bucket = "monk-client-images"
            key = data["path"].replace("s3:/monk-client-images/","")
            s3 = S3Path(bucket,key)
            im = PIL.Image.open(s3.download())
        else:
            im = PIL.Image.open(data["path"])
        

        bbox =  data["part_bbox"]
        img_crop = im.crop(bbox)
        img_crop = img_crop.resize(self.dim)
        img_crop = ((((np.array(img_crop)/255)*2)-1)).astype(np.float32)
        
        img_cutout = self.cutout(img_crop)

        return( img_crop,img_cutout )
        #return(np.array(img_crop).astype(np.float32))
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        IMGS,IMGS_CUT_OUT = self.__data_generation(list_IDs_temp)

        return IMGS,IMGS_CUT_OUT

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        IMGS = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        IMGS_CUT_OUT = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img,img_cutout=self.load_image(self.indexes[ID])
            IMGS[i,] = img
            IMGS_CUT_OUT[i,] = img_cutout

            

        IMGS= tf.convert_to_tensor(IMGS,dtype=tf.float32)
        IMGS_CUT_OUT= tf.convert_to_tensor(IMGS_CUT_OUT,dtype=tf.float32)       
        return IMGS,IMGS_CUT_OUT

def get_generator(json_paths,batch_size,size,damaged=False,
    n_holes=1,max_height=100,max_width=100,min_width = 50,min_height = 50,fill_value_mode = "zero"):
    
    generator = DataGenerator(json_paths,batch_size=batch_size,dim=(size,size),damaged=damaged,
    n_holes=n_holes,max_height=max_height,max_width=max_width,min_width = min_width,min_height = min_height,fill_value_mode = fill_value_mode)

    return(generator)