def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    from data.dataset_derain import DatasetDERAIN as D
    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
