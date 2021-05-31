import glob
import numpy as np
import PIL.Image as pil_image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class mvterDataset(Dataset):
	def __init__(self, o_root, r_root, rotate_gd, num_views=12, train=True):
		super(mvterDataset, self).__init__()

		self.classnames=[
							'airplane',
							#'bathtub',
							'bed','bench','bookshelf','bottle','bowl','car','chair',
							'cone','cup','curtain',
							#'desk',
							'door',
							#'dresser',
							'flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel',
							#'monitor','night_stand',
							'person','piano','plant','radio','range_hood','sink',
							#'sofa',
							'stairs','stool',
							#'table',
							'tent','toilet','tv_stand','vase','wardrobe','xbox'
						]
		self.o_root = o_root
		self.r_root = r_root
		self.num_views = num_views
		self.rotate_gd = rotate_gd

		self.o_filepaths = []
		self.r_filepaths = []

		group = 'train' if train else 'test'

		for cls in range(len(self.classnames)):
			for r_image_path in sorted(glob.glob('{}/{}/{}/*.png'.format(r_root, self.classnames[cls], group))):
				o_image_path = r_image_path.replace(self.r_root, self.o_root, 1)
				o_image_path = o_image_path.replace('_white', '.', 1)
				idxa = o_image_path.find('shaded_v')+8
				idxb = o_image_path.find('.png')
				o_image_path = o_image_path[:idxa]+format(int(o_image_path[idxa:idxb])+1,'03d')+'.png'

				self.o_filepaths.append(o_image_path)
				self.r_filepaths.append(r_image_path)

				self.transform = transforms.Compose([
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
				])

	def __getitem__(self, idx):
		o_path = self.o_filepaths[idx * self.num_views]
		r_path = self.r_filepaths[idx * self.num_views]
		class_name = o_path.split('/')[-3]
		class_id = self.classnames.index(class_name)

		key = 'ModelNet40/'+r_path[r_path.find(class_name):r_path.find('.obj')+4]
		euler = self.rotate_gd[key]

		# Use PIL instead
		o_imgs = []
		r_imgs = []
		for i in range(self.num_views):
			o_im = pil_image.open(self.o_filepaths[idx * self.num_views + i]).convert('RGB')
			r_im = pil_image.open(self.r_filepaths[idx * self.num_views + i]).convert('RGB')
			if self.transform:
				o_im = self.transform(o_im)
				r_im = self.transform(r_im)
			o_imgs.append(o_im)
			r_imgs.append(r_im)

		return (class_id, torch.tensor(euler), torch.stack(o_imgs), torch.stack(r_imgs), 
				self.o_filepaths[idx*self.num_views:(idx+1)*self.num_views],
				self.r_filepaths[idx*self.num_views:(idx+1)*self.num_views],)

	def __len__(self):
		return int(len(self.o_filepaths)/self.num_views)

if __name__ == '__main__':
	rotate_patch = np.load('../rotate_gd/rotate_patched.npy', allow_pickle=True)
	rotate_patch = rotate_patch.item()
	assert(len(rotate_patch) == 9449)
	train_dataset = mvterDataset('../rawdata/origin_12x', '../rawdata/rotate_12x', rotate_patch, train=True)
	test_dataset = mvterDataset('../rawdata/origin_12x', '../rawdata/rotate_12x', rotate_patch, train=False)
	print('trainset num: {}, testset num: {}'.format(len(train_dataset), len(test_dataset)))
	train_iter = DataLoader(dataset=train_dataset,
							batch_size=24,
							shuffle=True,
							pin_memory=True,
							drop_last=True)
	print('trainloader num of batches: {}'.format(len(train_iter)))
	label, euler, origin, rotate, o_filepath, r_filepath = iter(train_iter).next()
	print('origin batch shape: {},\nrotate batch shape: {},\nlabel batch shape: {}, euler angle batch shape: {}'.format(origin.shape, rotate.shape, label.shape, euler.shape))
	test_iter = DataLoader(dataset=test_dataset,
							batch_size=24,
							shuffle=True,
							pin_memory=True,
							drop_last=True)
	print('testloader num of batches: {}'.format(len(test_iter)))
	label, euler, origin, rotate, o_filepath, r_filepath = iter(test_iter).next()
	print('origin batch shape: {},\nrotate batch shape: {},\nlabel batch shape: {}, euler angle batch shape: {}'.format(origin.shape, rotate.shape, label.shape, euler.shape))


