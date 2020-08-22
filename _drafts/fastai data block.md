






fastai data block


# obj is a Pathlib object
def get_gt(obj): return im_gt_dict[obj.name]


src = ObjectItemList.from_folder(image_path)


src = src.label_from_func(get_gt)











getattr



ItemLists


First, it's going to check if self.train, which is an ImageImageList, has the attribute "label_from_folder"
- actually, it doesn't check, it just does it


then it checks to make sure that method is Callable, which it is

then it does the same thing for the validation data


then it defines another method called _inner
- we go into it with *args and **kwargs


then we run the ft function on the train data
- this is the label_from_folder method in the data_block api








