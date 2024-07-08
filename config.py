class Config(object):
    
    #==================================================================
    ### Face Detection 
    det_backbone = 'resnet50'   # Backbone network: mobile0.25, resnet50
    det_trained_weights = 'RetinaFace_Torch/weights/Resnet50_Final.pth'
    confidence_threshold = 0.02 
    top_k = 5000 
    nms_threshold = 0.4
    keep_top_k = 750
    vis_thres = 0.8   # Visualization threshold. 
    
    #==================================================================
    ### Face Recognition

    env = 'default'
    rec_backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    rec_trained_weights = "Arcface_Torch/checkpoint/resnet18_110.pth"
    save_interval = 10

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'