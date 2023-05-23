import os

def run_exp(shot):
    query = 15
    way = 5
    gpu = 1
    dataname = 'mini'    # mini tiered cub cifar_fs
    modelname = 'res12'  # res12 wrn28
    the_command = 'python3.6 main.py' \
        + ' --addVal=' + str(1) \
        + ' --model_type=' + modelname \
        + ' --dataset=' + dataname \
        + ' --lr=' + str(0.0001) \
        + ' --shot=' + str(shot) \
        + ' --way=' + str(way) \
        + ' --max_epoch=' + str(10) \
        + ' --train_query=' + str(query) \
        + ' --val_query=' + str(query) \
        + ' --gpu=' + str(gpu) \
        + ' --opt=' + 'SGD' \
        + ' --metric=' + 'cos' \
        + ' --dataset_dir=' + '/data/FSL/features/' + dataname + '/' + modelname

    os.system(the_command + ' --phase=train')
    os.system(the_command + ' --phase=test')


run_exp(shot=1)
run_exp(shot=5)