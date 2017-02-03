from my_utils_feat import get_data, save_array, Adam, get_batches, batch_generator_train
import os
from vgg16bn import Vgg16BN

batch_size=8
path = '../input/stage1/'
# trn = batch_generator_train(path, shuffle=False, batch_size=batch_size, class_mode=None, target_size=(512,512))
# batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
# val = get_batches(path+'valid', shuffle=False, batch_size=batch_size, class_mode=None, target_size=(360,640))

if not os.path.exists(path.split('stage1')[0] + 'results'):
    os.makedirs(path.split('stage1')[0] + 'results')
import glob
import pandas as pd

# save_array(path+'results/trn_640.dat', trn)
# save_array(path+'results/val_640.dat', val)
# save_array(path+'results/test_640.dat', test)
nb_train= len(glob.glob(path+'*/*.dcm'))
# nb_train_samples = 500
# nb_val= len(glob.glob(path+'valid/*/*.jpg'))
# nb_test= len(glob.glob('input/test/*/*.jpg'))

vgg512 = Vgg16BN((512, 512), use_preprocess=True).model
vgg512.pop()
print(vgg512.input_shape, vgg512.output_shape)
vgg512.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

train_csv_table = pd.read_csv('../input/stage1_labels.csv')

print(len([x[0] for x in os.walk(path)][1:]))
print(len(glob.glob(path+'*/*.dcm')))
patient_ids = [x[0].split('/')[-1] for x in os.walk(path)][1:]

for an_index,an_id in enumerate(patient_ids):
    if an_index %100==0:
        print('%s done on %s total' %(an_index, len(patient_ids)))
    # if not os.path.exists(path.split('stage1')[0] + 'results'+str(an_id)):
    #     os.makedirs(path.split('stage1')[0] + 'results'+str(an_id))
    if os.path.exists(path.split('stage1')[0] + 'results/conv_ft_512_'+str(an_id)+'.dat'):
        print('%s already converted' % an_id)
        continue
    images_patient = glob.glob(path + an_id + '/*.dcm')

    trn = batch_generator_train(images_patient, train_csv_table, batch_size=batch_size, from_gray_to_rgb=True)

    conv_trn_feat = vgg512.predict_generator(trn, len(images_patient))

    save_array(path.split('stage1')[0] + 'results/conv_ft_512_'+str(an_id)+'.dat', conv_trn_feat)

    print(conv_trn_feat.shape, len(images_patient))
    del conv_trn_feat



#
# # conv_val_feat = vgg640.predict_generator(val,nb_val)
# # save_array(path.split('stage1')[0] + 'results/conv_val_640.dat', conv_val_feat)
# #
# # del conv_val_feat






# # test = get_data('input/test', (360,640))
# test = get_batches('input/test', shuffle=False, batch_size=batch_size, class_mode=None, target_size=(360,640))
#
# conv_test_feat = vgg640.predict_generator(test,nb_test)
#
# save_array(path+'results/conv_test_640.dat', conv_test_feat)

# conv_val_feat = load_array(path+'results/conv_val_640.dat')
# conv_trn_feat = load_array(path+'results/conv_trn_640.dat')
