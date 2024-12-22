def define_Model(opt):
    from models.model_plain import ModelPlain as M
    m = M(opt)
    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
