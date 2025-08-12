import mmcv

def load_annotations(anno_file_diff, anno_file_ii):
    # merge Load annotations
    data_diff = mmcv.load(anno_file_diff, file_format='pkl')
    data_infos_diff = list(sorted(data_diff['infos'],key=lambda e:e['timestamp']))
    data_ii = mmcv.load(anno_file_ii, file_format='pkl')
    data_infos_ii = list(sorted(data_ii['infos'],key=lambda e:e['timestamp']))
    data_infos = data_infos_diff
    for i, d in enumerate(data_infos_ii):
        if d not in data_infos:
            data_infos.append(d)
        else:
            if d[i]['token']!=data_infos_ii[i]['token']:
                assert False, "same key but different value"
    data = {'metadata':{'version':'v1.0-trainval'},'infos': data_infos}
    mmcv.dump(data, 'merged_nuscenes_infos_val.pkl')

def check_nuscenes_infos(data_path):
    data = mmcv.load(data_path)
    print(len(data['infos']))
    print(data['metadata'])

if __name__ == '__main__':
    
    anno_file_diff = '/home/user/waytous/cjj/DiffusionOcc/data/infos/nuscenes_infos_val.pkl'
    anno_file_ii = '/home/user/waytous/cjj/DiffusionOcc/data/nuscenes/world-nuscenes_infos_val.pkl'
    load_annotations(anno_file_diff,anno_file_ii)
    # check_nuscenes_infos('/home/user/waytous/cjj/DiffusionOcc/merged_nuscenes_infos_train.pkl')