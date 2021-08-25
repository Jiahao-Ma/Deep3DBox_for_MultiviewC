class Config(object):
    root = r'F:\ANU\ENGN8602\Data\MultiviewC_dataset'
    split = 0.9
    bins = 2

    batch_size = 32
    num_workers = 4

    # Loss_theta = L_conf + W * L_loc
    # L = alpha * L_dim + L_theta
    alpha = 0.6
    W = 0.4

    model_path = r'weights\deep3dbox.pkl'
    load_model = True

    faster_rcnn_path = r'weights\fasterrcnn'

    min_size = 600
    max_size = 1000



    epochs = 20

    lr=0.0001 #0.0001
    momentum=0.9

opt = Config()