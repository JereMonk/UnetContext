import numpy as np
from monk import BBox
import tensorflow as tf
from monk import Dataset

def my_to_bbox(polygon, allow_unsafe=False):
        """ Get the smallest BBox encompassing the polygon """
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf
        for subpol in polygon.points:
            xmin = min(xmin, *subpol[:, 0])
            xmax = max(xmax, *subpol[:, 0])
            ymin = min(ymin, *subpol[:, 1])
            ymax = max(ymax, *subpol[:, 1])
            
        xmin = max(0,xmin)
        ymin = max(0,ymin)

        xmax= min(polygon.image_size[0],xmax)
        ymax = min(polygon.image_size[1],ymax)

        
        return BBox(
            label=polygon.label,
            image_size=polygon.image_size,
            xyxy=[xmin, ymin, xmax, ymax],
            allow_unsafe=allow_unsafe,
            attributes=polygon.attributes.copy(),
        )

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset, batch_size=32, dim=(128,128), n_channels=3,shuffle=True):
          
        self.dataset=dataset
        self.dim = dim ###
        self.batch_size = batch_size  ##
        self.list_IDs = np.arange(len(dataset)) ###
        self.n_channels = n_channels ##
        self.shuffle = shuffle ##
        
        self.get_map_id()
        self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        i=0
        map_id ={}
        for _,imds in enumerate(self.dataset):
            for ann_id,ann in enumerate(imds.anns["polygons"]):
                map_id[i]=[imds.id,ann_id]
                i+=1
        self.map_id = map_id
        
    def load_image(self,ids):
        imds = self.dataset[ids[0]]
        ann = imds.anns["polygons"][ids[1]]
        img_crop = imds.image.crop(my_to_bbox(ann)).resize(self.dim)
        
        #return(img_crop.rgb)
        return(((img_crop.rgb/255)*2)-1)
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
       

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            

            
            try:
                X[i,] = self.load_image(self.map_id[ID])

            except Exception as e:
                X[i,] = np.zeros((*self.dim, self.n_channels))
                print(str(e))

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return tf.convert_to_tensor(X,dtype=tf.float32)


def get_generator(path,size,batch_size,to_keep):

    dataset_parts = Dataset.from_coco(path,"")
    dataset_filtered = dataset_parts.filter_images_with_cats(keep=to_keep).filter_cats(keep=to_keep)
    generator = DataGenerator(dataset_filtered,batch_size=batch_size,dim=(size,size))

    return(generator)


