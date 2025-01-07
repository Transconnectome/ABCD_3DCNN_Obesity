import os
from os import listdir
from os.path import isfile, join
import glob


from utils.utils import case_control_count
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num

import random
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor, RandAxisFlip, RandCoarseDropout, RandSpatialCrop, SpatialCrop, CenterSpatialCrop, RandAffine
from monai.data import ImageDataset

def check_study_sample(study_sample):
    elif study_sample == 'ABCD_1y_after_become_overweight':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/5.demo_qc/BMI_prediction/ABCD_phenotype_total_1years_become_overweight_10PS_stratified_partitioned_5fold.csv'
    elif study_sample == 'ABCD_2y_after_become_overweight':
        image_dir = '/scratch/connectome/3DCNN/data/1.ABCD/1.1.sMRI_MNI_warped'
        phenotype_dir = '/scratch/connectome/3DCNN/data/1.ABCD/5.demo_qc/BMI_prediction/ABCD_phenotype_total_2years_become_overweight_10PS_stratified_partitioned_5fold.csv'   
 return image_dir, phenotype_dir 


def loading_images(image_dir, args, study_sample='UKB'):
    if study_sample.find('UKB') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    elif study_sample.find('ABCD') != -1:
        image_files = glob.glob(os.path.join(image_dir,'*.nii.gz'))
    image_files = sorted(image_files)
   
    #image_files = image_files[:1000]
    print("Loading image file names as list is completed")
    return image_files


def loading_phenotype(phenotype_dir, args, study_sample='UKB', undersampling_dataset_target=None, partitioned_dataset_number=None):
    if study_sample.find('UKB') != -1:
        subject_id_col = 'eid'
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'

    targets = args.cat_target + args.num_target
    if undersampling_dataset_target:
        if not undersampling_dataset_target in targets: 
            col_list = targets + [subject_id_col] + [undersampling_dataset_target]
        else: 
            col_list = targets + [subject_id_col]
    elif partitioned_dataset_number is not None: 
        assert undersampling_dataset_target is None 
        col_list = targets + [subject_id_col] + ["partition%s" % partitioned_dataset_number]
    else: 
        col_list = targets + [subject_id_col]
    ### get subject ID and target variables
    subject_data = pd.read_csv(phenotype_dir)
    subject_data = subject_data.loc[:,col_list]
    subject_data = subject_data.sort_values(by=subject_id_col)
    subject_data = subject_data.dropna(axis = 0)
    subject_data = subject_data.reset_index(drop=True) # removing subject have NA values in sex

    
    ### preprocessing categorical variables and numerical variables
    if args.cat_target:
        subject_data = preprocessing_cat(subject_data, args)
        #num_classes = int(subject_data[args.cat_target].nunique().values)
    #if args.num_target:
        #subject_data = preprocessing_num(subject_data, args)
        #num_classes = 1 
    
    return subject_data, targets


## combine categorical + numeric
def combining_image_target(subject_data, image_files, target_list, undersampling_dataset_target=None,partitioned_dataset_number=None ,study_sample='UKB'):
    if study_sample.find('UKB') !=- 1:
        subject_id_col = 'eid'
        suffix_len = -7
    elif study_sample.find('ABCD') != -1:
        subject_id_col = 'subjectkey'
        suffix_len = -7
    imageFiles_labels = []
    
    subj = []
    if type(subject_data[subject_id_col][0]) == np.str_ or type(subject_data[subject_id_col][0]) == str:
        for i in range(len(image_files)):
            subject_id = os.path.split(image_files[i])[-1]
            subj.append(str(subject_id[:suffix_len]))
    elif type(subject_data[subject_id_col][0]) == np.int_ or type(subject_data[subject_id_col][0]) == int:
        for i in range(len(image_files)):
            subject_id = os.path.split(image_files[i])[-1]
            subj.append(int(subject_id[:suffix_len]))

    image_list = pd.DataFrame({subject_id_col:subj, 'image_files': image_files})
    subject_data = pd.merge(subject_data, image_list, how='inner', on=subject_id_col)
    subject_data = subject_data.sort_values(by=subject_id_col)

    if undersampling_dataset_target:
        if not undersampling_dataset_target in target_list:
            col_list = target_list + [undersampling_dataset_target] + ['image_files']
        else:
            col_list = target_list + ['image_files']
    elif partitioned_dataset_number is not None:
        assert undersampling_dataset_target is None 
        col_list = target_list + ["partition%s" % partitioned_dataset_number] + ['image_files']

    else: 
        col_list = target_list + ['image_files']
    
    for i in tqdm(range(len(subject_data))):
        imageFile_label = {}
        for j, col in enumerate(col_list):
            imageFile_label[col] = subject_data[col][i]
        imageFiles_labels.append(imageFile_label)
        
    return imageFiles_labels



# defining train,val, test set splitting function
def partition_dataset(imageFiles_labels, targets, args):
    #random.shuffle(imageFiles_labels)

    images = []
    labels = []

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        images.append(image)
        labels.append(label)
    


    resize = tuple(args.resize)
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                               RandRotate90(prob=0.3),
                               RandAxisFlip(prob=0.3),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)

    # image and label information of train
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image and label information of valid
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    # image and label information of test
    images_test = images[num_train+num_val:]
    labels_test = labels[num_train+num_val:]

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("Training Samples: {}. Valdiation Samples: {}. Test Samples: {}". format(len(labels_train), len(labels_val), len(labels_test)))

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition
## ====================================== ##


def matching_partition_dataset(imageFiles_labels, reference_dataset, targets, args):
    #random.shuffle(imageFiles_labels)

    images_train = []
    labels_train = []
    images_val = []
    labels_val = []
    images_test = []
    labels_test = []
    

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if image in reference_dataset['train'].image_files: 
            images_train.append(image)
            labels_train.append(label)
        elif image in reference_dataset['val'].image_files: 
            images_val.append(image)
            labels_val.append(label)
        elif image in reference_dataset['test'].image_files: 
            images_test.append(image)
            labels_test.append(label)
    


    resize = tuple(args.resize)
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])
    """
    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)

    # image and label information of train
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image and label information of valid
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    # image and label information of test
    images_test = images[num_train+num_val:]
    labels_test = labels[num_train+num_val:]
    """
    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("The number of Training samples: {}. The number of Validation samples: {}. The number of Test samples: {}".format(len(labels_train), len(labels_val), len(labels_test)))

    return partition
## ====================================== ##


def undersampling_ALLset(imageFiles_labels, targets, undersampling_dataset_target, args):
    #random.shuffle(imageFiles_labels)

    total_num_control = 0 
    total_num_case = 0 
    for imageFile_label in imageFiles_labels:
        if imageFile_label[undersampling_dataset_target] == 0: 
            total_num_control += 1 
        elif imageFile_label[undersampling_dataset_target] == 1: 
            total_num_case += 1 

    num_case_train = int(total_num_case*args.train_size)
    num_control_train = num_case_train 
    num_case_test = int(total_num_case*args.test_size)
    num_control_test = num_case_test
    num_case_val = total_num_case - num_case_train - num_case_test
    num_control_val = num_case_val
    

    # dataset list for train, validation, and test 
    images_case_train = []
    labels_case_train = [] 
    images_control_train = []
    labels_control_train = [] 

    images_case_test = []
    labels_case_test = [] 
    images_control_test = []
    labels_control_test = [] 

    images_case_val = []
    labels_case_val = [] 
    images_control_val = []
    labels_control_val = [] 

    count_case_train = 0 
    count_control_train = 0    
    count_case_test = 0 
    count_control_test = 0
    count_case_val = 0 
    count_control_val = 0 



    for imageFile_label in imageFiles_labels: 
        image = imageFile_label['image_files']
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if imageFile_label[undersampling_dataset_target] == 0: 
            if count_control_test < num_control_test: 
                images_control_test.append(image)
                labels_control_test.append(label)
                count_control_test += 1 
            else:
                if count_control_train < num_control_train:
                    images_control_train.append(image)
                    labels_control_train.append(label)     
                    count_control_train += 1               
                else:
                    if count_control_val < num_control_val:
                        images_control_val.append(image)
                        labels_control_val.append(label)
                        count_control_val += 1 

        elif imageFile_label[undersampling_dataset_target] == 1: 
            if count_case_test < num_case_test: 
                images_case_test.append(image)
                labels_case_test.append(label)   
                count_case_test += 1 
            else: 
                if count_case_train < num_case_train:
                    images_case_train.append(image)
                    labels_case_train.append(label)
                    count_case_train += 1 
                else:
                    if count_case_val < num_case_val: 
                        images_case_val.append(image)
                        labels_case_val.append(label)  
                        count_case_val += 1 

    
    resize = tuple(args.resize)
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    images_test = images_control_test + images_case_test 
    labels_test = labels_control_test + labels_case_test 

    images_train = images_control_train + images_case_train
    labels_train = labels_control_train + labels_case_train

    images_val = images_control_val + images_case_val
    labels_val = labels_control_val + labels_case_val

    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("Training Samples: {}. Valdiation Samples: {}. Test Samples: {}". format(len(labels_train), len(labels_val), len(labels_test)))

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition


def matching_undersampling_ALLset(imageFiles_labels, reference_dataset,  undersampling_dataset_target, args):
    #random.shuffle(imageFiles_labels)
    targets = args.cat_target + args.num_target

    total_num_control = 0 
    total_num_case = 0 
    for imageFile_label in imageFiles_labels:
        if imageFile_label[undersampling_dataset_target] == 0: 
            total_num_control += 1 
        elif imageFile_label[undersampling_dataset_target] == 1: 
            total_num_case += 1 

    num_case_train = int(total_num_case*args.train_size)
    num_control_train = num_case_train 
    num_case_test = int(total_num_case*args.test_size)
    num_control_test = num_case_test
    num_case_val = total_num_case - num_case_train - num_case_test
    num_control_val = num_case_val

    # dataset list for train, validation, and test 
    images_case_train = []
    labels_case_train = [] 
    images_control_train = []
    labels_control_train = [] 

    images_case_test = []
    labels_case_test = [] 
    images_control_test = []
    labels_control_test = [] 

    images_case_val = []
    labels_case_val = [] 
    images_control_val = []
    labels_control_val = [] 

    count_case_train = 0 
    count_control_train = 0    
    count_case_test = 0 
    count_control_test = 0
    count_case_val = 0 
    count_control_val = 0 

    for imageFile_label in imageFiles_labels: 
        image = imageFile_label['image_files']
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if imageFile_label[undersampling_dataset_target] == 0: 
            if count_control_test < num_control_test:
                if image in reference_dataset['test'].image_files:  
                    images_control_test.append(image)
                    labels_control_test.append(label)
                count_control_test += 1 
            else:
                if count_control_train < num_control_train:
                    if image in reference_dataset['train'].image_files: 
                        images_control_train.append(image)
                        labels_control_train.append(label)     
                    count_control_train += 1               
                else:
                    if count_control_val < num_control_val:
                        if image in reference_dataset['val'].image_files: 
                            images_control_val.append(image)
                            labels_control_val.append(label)
                        count_control_val += 1 

        elif imageFile_label[undersampling_dataset_target] == 1: 
            if count_case_test < num_case_test:
                if image in reference_dataset['test'].image_files:  
                    images_case_test.append(image)
                    labels_case_test.append(label)   
                count_case_test += 1 
            else: 
                if count_case_train < num_case_train:
                    if image in reference_dataset['train'].image_files: 
                        images_case_train.append(image)
                        labels_case_train.append(label)
                    count_case_train += 1 
                else:
                    if count_case_val < num_case_val: 
                        if image in reference_dataset['val'].image_files: 
                            images_case_val.append(image)
                            labels_case_val.append(label)  
                        count_case_val += 1 
    


    resize = tuple(args.resize)
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])
    """
    # number of total / train,val, test
    num_total = len(images)
    num_train = int(num_total*(1 - args.val_size - args.test_size))
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)

    # image and label information of train
    images_train = images[:num_train]
    labels_train = labels[:num_train]

    # image and label information of valid
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    # image and label information of test
    images_test = images[num_train+num_val:]
    labels_test = labels[num_train+num_val:]
    """

    images_test = images_control_test + images_case_test 
    labels_test = labels_control_test + labels_case_test 

    images_train = images_control_train + images_case_train
    labels_train = labels_control_train + labels_case_train

    images_val = images_control_val + images_case_val
    labels_val = labels_control_val + labels_case_val


    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("The number of Training samples: {}. The number of Validation samples: {}. The number of Test samples: {}".format(len(labels_train), len(labels_val), len(labels_test)))

    return partition
## ====================================== ##


# defining train,val, test set splitting function
def partition_dataset_predefined(imageFiles_labels, targets, partitioned_dataset_number, args):
    #random.shuffle(imageFiles_labels)

    images_train = []
    labels_train = []
    images_val = []
    labels_val = []
    images_test = []
    labels_test = []

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['image_files']
        label = {}
        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        if imageFile_label['partition%s' % partitioned_dataset_number] =='train': 
            images_train.append(image)
            labels_train.append(label)
        elif imageFile_label['partition%s' % partitioned_dataset_number] =='val': 
            images_val.append(image)
            labels_val.append(label)    
        elif imageFile_label['partition%s' % partitioned_dataset_number] =='test': 
            images_test.append(image)
            labels_test.append(label)

    resize = tuple(args.resize)
    roi_center = (90, 110, 80)
    roi_size = (150, 185, 160)
    #cutout_size = (8, 8, 8)
    #cutout_ratio = 0.2
    #num_holes = int(cutout_ratio * ((resize[0] // cutout_size[0]) * (resize[1] // cutout_size[1]) * (resize[2] // cutout_size[2])))
    cutout_size = (32, 32, 32)
    num_holes = 1 
    train_transform = Compose([ScaleIntensity(),    
#                               SpatialCrop(roi_center=roi_center, roi_size=roi_size),                           
                               AddChannel(),
                               Resize(resize),
#                               RandSpatialCrop(roi_size=resize, random_size=False),
#                               RandCoarseDropout(holes=num_holes,spatial_size=cutout_size, prob=0.5),
#                               RandCoarseDropout(holes=num_holes,spatial_size=cutout_size, fill_value=0,prob=0.5),
#                               CenterSpatialCrop(roi_size=resize),
                               RandRotate90(prob=0.5),
                               RandAxisFlip(prob=0.5),
#                               RandAffine(prob=0.5, padding_mode='zeros', translate_range=(int(resize[0]*0.1),)*3, rotate_range=(np.pi/36,)*3, spatial_size=resize,cache_grid=True),
                               ToTensor()])

    val_transform = Compose([ScaleIntensity(),
#                               SpatialCrop(roi_center=roi_center, roi_size=roi_size),   
                             AddChannel(),
                               Resize(resize),
#                             CenterSpatialCrop(roi_size=resize),
                             ToTensor()])

    test_transform = Compose([ScaleIntensity(),
#                               SpatialCrop(roi_center=roi_center, roi_size=roi_size),   
                              AddChannel(),
                               Resize(resize),
#                              CenterSpatialCrop(roi_size=resize),
                              ToTensor()])
    
    if args.finetune_undersample:
        np.random.seed(args.seed)
        ### Training samples
        # extract index 
        labels_train_tmp = []
        for i in range(len(labels_train)): 
            labels_train_tmp.append(labels_train[i][targets[0]])
        case_train_index = np.where(np.array(labels_train_tmp) == 1)[0]
        control_train_index = np.where(np.array(labels_train_tmp) == 0)[0]
        # undersample
        train_random_index = np.random.randint(0,len(case_train_index),len(case_train_index))
        control_train_index = control_train_index[train_random_index]
        # reassigning samples
        images_train_case = np.array(images_train)[case_train_index].tolist()
        images_train_control = np.array(images_train)[control_train_index].tolist()
        labels_train_case = np.array(labels_train)[case_train_index].tolist()
        labels_train_control = np.array(labels_train)[control_train_index].tolist()
        images_train = images_train_case + images_train_control
        labels_train = labels_train_case + labels_train_control
        
        ### Validation samples
        # extract index 
        labels_val_tmp = []
        for i in range(len(labels_val)): 
            labels_val_tmp.append(labels_val[i][targets[0]])
        case_val_index = np.where(np.array(labels_val_tmp) == 1)[0]
        control_val_index = np.where(np.array(labels_val_tmp) == 0)[0]
        # undersample
        train_random_index = np.random.randint(0,len(case_val_index),len(case_val_index))
        control_val_index = control_val_index[train_random_index]
        # reassigning samples
        images_val_case = np.array(images_val)[case_val_index].tolist()
        images_val_control = np.array(images_val)[control_val_index].tolist()
        labels_val_case = np.array(labels_val)[case_val_index].tolist()
        labels_val_control = np.array(labels_val)[control_val_index].tolist()
        images_val = images_val_case + images_val_control
        labels_val = labels_val_case + labels_val_control

        ### Test samples
        # extract index 
        labels_test_tmp = []
        for i in range(len(labels_test)): 
            labels_test_tmp.append(labels_test[i][targets[0]])
        case_test_index = np.where(np.array(labels_test_tmp) == 1)[0]
        control_test_index = np.where(np.array(labels_test_tmp) == 0)[0]
        # undersample
        train_random_index = np.random.randint(0,len(case_test_index),len(case_test_index))
        control_test_index = control_test_index[train_random_index]
        # reassigning samples
        images_test_case = np.array(images_test)[case_test_index].tolist()
        images_test_control = np.array(images_test)[control_test_index].tolist()
        labels_test_case = np.array(labels_test)[case_test_index].tolist()
        labels_test_control = np.array(labels_test)[control_test_index].tolist()
        images_test = images_test_case + images_test_control
        labels_test = labels_test_case + labels_test_control


    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    print("Training Samples: {}. Valdiation Samples: {}. Test Samples: {}". format(len(labels_train), len(labels_val), len(labels_test)))

    case_control_count(labels_train, 'train', args)
    case_control_count(labels_val, 'validation', args)
    case_control_count(labels_test, 'test', args)

    return partition
