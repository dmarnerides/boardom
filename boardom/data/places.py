import os
from torch.utils.data import Dataset
import boardom as bd


class PlacesDataset(Dataset):
    def __init__(
        self,
        data_root_path,
        only_color=False,
        mode='training',
        preprocess=None,
        load_fn=None,
        need_labels=True,
    ):
        super().__init__()
        modes = {
            'training': 'train_standard',
            'validation': 'val',
            'testing': 'test',
        }
        assert mode in modes.keys()
        data_root_path = bd.process_path(data_root_path)
        flist_root = os.path.join(data_root_path, mode)
        txt_file_list = 'places365_{0}{1}.txt'.format(
            modes[mode], '_color' if only_color else ''
        )
        txt_file_list = os.path.join(data_root_path, txt_file_list)

        with open(txt_file_list) as file:
            contents = file.readlines()

        self.need_labels = need_labels

        if mode == 'testing':
            flist = [x.strip().split(' ')[0] for x in contents]
            self.file_list = [os.path.join(flist_root, x) for x in flist]
            self.need_labels = False
        if mode == 'validation':
            flist, labels = zip(*[x.strip().split(' ') for x in contents])
            self.file_list = [os.path.join(flist_root, x) for x in flist]
            self.labels = [int(x) for x in labels]
        else:
            flist, labels = zip(*[x.strip().split(' ') for x in contents])
            self.file_list = [os.path.join(flist_root, x[1:]) for x in flist]
            self.labels = [int(x) for x in labels]

        class_name_list = os.path.join(data_root_path, 'categories_places365.txt')
        with open(class_name_list) as file:
            contents = file.readlines()

        cls_ind = [x.strip().split(' ') for x in contents]
        cls_names, indices = zip(*cls_ind)
        cls_names = [x[3:] for x in cls_names]
        indices = [int(x) for x in indices]
        self.cls_to_ind = {c: l for c, l in zip(cls_names, indices)}
        self.ind_to_cls = {str(l): c for c, l in zip(cls_names, indices)}

        self.preprocess = preprocess
        self.load = bd.imread if load_fn is None else load_fn

    def get_class_name(self, i_label):
        return self.ind_to_cls[str(i_label)]

    def get_label_index(self, class_name):
        return self.cls_to_ind[class_name]

    def __getitem__(self, index):
        img = self.load(self.file_list[index])
        if self.preprocess is not None:
            img = self.preprocess(img)
        if self.need_labels:
            return img, self.labels[index]
        else:
            return img

    def __len__(self):
        return len(self.file_list)
