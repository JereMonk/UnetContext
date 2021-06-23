import tensorflow as tf
import numpy as np
from monk import BBox, Dataset
from monk import Classes

### DAMAGED

class DamagedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset, batch_size=32, dim=(128,128), n_channels=3,shuffle=True,to_keep=[],area_threshold=5000):
          
        self.dataset=dataset
        self.dim = dim ###
        self.batch_size = batch_size  ##
        #self.list_IDs = np.arange(len(dataset)) ###
        self.n_channels = n_channels ##
        self.shuffle = shuffle ##
        self.to_keep=to_keep
        self.area_threshold=area_threshold

        
        self.get_map_id()
        #self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        map_id ={}
        i=0

        for _,imds in enumerate(self.dataset):

            parts=[]
            to_keep = []

            for ind,poly in enumerate(imds.anns['polygons']):
                att = poly.attributes
                label =att["part_label"]
                if (not label in parts and label in self.to_keep and poly.area>self.area_threshold):
                    to_keep.append(ind)
                    parts.append(label)

            for ind in to_keep:
                map_id[i]=[imds.id,ind]
                i+=1
        
        self.list_IDs = np.arange(len(map_id))
        self.indexes = np.arange(len(map_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        self.map_id = map_id
        
    def load_image(self,ids):
        imds = self.dataset[ids[0]]
        ann = imds.anns["polygons"][ids[1]]
        att =ann.attributes
        label =att["part_label"]
       
        img_crop = imds.image.crop(BBox(xyxy=[att["x1_part"],att["y1_part"],att["x2_part"],att["y2_part"],])).resize(self.dim)
       
        img = np.array(img_crop.rgb/255,dtype=np.dtype('float32'))
        
        return(((img)*2)-1)
        
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
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X


def concat_dataset(datasets, keep_id=False):
    all_images = []
    all_classes = {}
    all_tags = {}
    for dataset in datasets:
        if not dataset:
            continue

        all_images.extend(dataset)
        all_classes.update({x["name"]: x for x in dataset.classes.iter_objects()})
        all_tags.update({x["name"]: x for x in dataset.tags.iter_objects()})

    all_classes_obj = list(all_classes.values())
    all_tags_obj = list(all_tags.values())

    if not keep_id:
        i = 1
        for ctype in [all_classes_obj, all_tags_obj]:
            for obj in ctype:
                obj["id"] = i
                i += 1

    dataset = Dataset(classes=Classes(all_classes_obj), tags=Classes(all_tags_obj))
    for imdb in all_images:
        dataset.add_image(imdb, force_new_id=True)

    return dataset

def get_damaged_generator(paths, batch_size=3, dim=(128,128), to_keep=[],area_threshold=5000):

    ds = []
    for path in paths:
        dataset = Dataset.from_coco(path,"")
        ds.append(dataset)
    dataset = concat_dataset(ds)
    gen = DamagedDataGenerator(dataset,batch_size=batch_size,dim=dim,to_keep=to_keep,area_threshold=area_threshold)

    return(gen)


## NON_DAMAGED



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

class NonDamagedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset, batch_size=3, dim=(128,128), n_channels=3,shuffle=True,area_threshold=5000):
          
        self.dataset=dataset
        self.dim = dim ###
        self.batch_size = batch_size  ##
        #self.list_IDs = np.arange(len(dataset)) ###
        self.n_channels = n_channels ##
        self.shuffle = shuffle ##
        self.area_threshold=area_threshold
        self.get_map_id()
        #self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        i=0
        map_id ={}
        for _,imds in enumerate(self.dataset):
            for poly_id,poly in enumerate(imds.anns["polygons"]):
                
                if(poly.area>self.area_threshold):
                    map_id[i]=[imds.id,poly_id]
                    i+=1
               
        self.map_id = map_id

        self.list_IDs = np.arange(len(map_id))
        self.indexes = np.arange(len(map_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_image(self,ids):
        imds = self.dataset[ids[0]]
        ann = imds.anns["polygons"][ids[1]]
        att =ann.attributes
        img_crop = imds.image.crop(my_to_bbox(ann)).resize(self.dim)
       
        img = np.array(img_crop.rgb/255,dtype=np.dtype('float32'))
        
        return(((img)*2)-1)

        return ( img )
        
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
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X

def get_non_damaged_generator(paths, batch_size=3, dim=(128,128), to_keep=[],area_threshold=5000):


    ds = []
    for path in paths:
        dataset = Dataset.from_coco(path,"")
        ds.append(dataset)
    dataset = concat_dataset(ds)
    dataset_non_damaged = dataset.filter_images_with_cats(keep=to_keep).filter_cats(keep=to_keep)
    gen = NonDamagedDataGenerator(dataset_non_damaged,batch_size=batch_size,dim=dim,area_threshold=area_threshold)

    return(gen)

## MIXED

class MixedDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,dataset_damages,dataset_parts, batch_size=32, dim=(128,128), n_channels=3,shuffle=True,to_keep=[],area_threshold=2000):
          
        self.dataset_parts=dataset_parts
        self.dataset_damages=dataset_damages
        self.dim = dim 
        self.batch_size = batch_size  
        #self.list_IDs = np.arange(len(dataset_damages)+len(dataset_parts)) 
        self.n_channels = n_channels 
        self.shuffle = shuffle 
        self.to_keep =to_keep
        self.area_threshold=area_threshold

    

        self.get_map_id()
        #self.on_epoch_end()
        
        #self.labels = labels
        #self.n_classes = n_classes
        
    def get_map_id(self):
        i=0
        map_id_1 ={}
        
        for _,imds in enumerate(self.dataset_damages):
            parts=[]
            to_keep = []

            for ind,poly in enumerate(imds.anns['polygons']):
                att = poly.attributes

                label =att["part_label"]
                if self.to_keep =='all':
                    if (not label in parts and poly.area>self.area_threshold):
                        to_keep.append(ind)
                        parts.append(label)
                else:
                    if (not label in parts and label in self.to_keep and poly.area>self.area_threshold):
                        to_keep.append(ind)
                        parts.append(label)

            for ind in to_keep:
                map_id_1[i]=[1,imds.id,ind]
                i+=1
        i=0
        map_id_0 ={}
        for _,imds in enumerate(self.dataset_parts):
            for poly_id,poly in enumerate(imds.anns["polygons"]):
                
                if(poly.area>self.area_threshold):
                    map_id_0[i]=[0,imds.id,poly_id]
                    i+=1
        ## BALANCE 
        map_id ={}
        i=0
        for j in range(0,min(len(map_id_0),len(map_id_1))):
            map_id[i]=map_id_0[j]
            i+=1

        for j in range(0,min(len(map_id_0),len(map_id_1))):
            map_id[i]=map_id_1[j]
            i+=1

        self.map_id = map_id

        self.list_IDs = np.arange(len(map_id)) 
        self.indexes = np.arange(len(map_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def load_image(self,ids):

        if ids[0]==1:
            

            
            imds = self.dataset_damages[ids[1]]
            poly = imds.anns["polygons"][ids[2]]
            
            att = poly.attributes
            label =att["part_label"]
            
    
            img_crop = imds.image.crop(BBox(xyxy=[att["x1_part"],att["y1_part"],att["x2_part"],att["y2_part"],])).resize(self.dim)

            
        
        if ids[0]==0:
          
            imds = self.dataset_parts[ids[1]]
            poly = imds.anns["polygons"][ids[2]]

            att =poly.attributes
     


            img_crop = imds.image.crop(my_to_bbox(poly)).resize(self.dim)
        
        
        
        img = np.array(img_crop.rgb/255,dtype=np.dtype('float32'))
        
        return(((img)*2)-1)
        
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
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
   
            X[i,] = self.load_image(self.map_id[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return tf.convert_to_tensor(X,dtype=tf.float32)

def get_mixed_generator(paths_damaged,path_non_damaged, batch_size=3, dim=(128,128), to_keep='all',area_threshold=2000):
    
    ds = []
    for path in paths_damaged:
        dataset = Dataset.from_coco(path,"")
        ds.append(dataset)
    dataset_damaged = concat_dataset(ds)

    dataset_non_damaged = Dataset.from_coco(path_non_damaged,'')
    if (to_keep!='all'):
        dataset_non_damaged = dataset_non_damaged.filter_images_with_cats(keep=to_keep).filter_cats(keep=to_keep)
    gen = MixedDataGenerator(dataset_damages=dataset_damaged,dataset_parts=dataset_non_damaged,batch_size=batch_size,dim=dim,to_keep=to_keep,area_threshold=area_threshold)
    return(gen)
