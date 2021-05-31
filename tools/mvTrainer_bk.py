import torch
import torch.nn as nn
import torch.nn.functional as F

class mvterTrainer():
	def __init__(self, log_dir, model, train_loader, test_loader, optimizer, scheduler, num_views=12, w=1.0):
		self.log_dir = log_dir
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.num_views = num_views
		self.w = w

	def train(self, epoch_start, epochs):
		results = {'train_loss': [], 'test_acc@1': []}
		model = nn.DataParallel(self.model).cuda()

		for epoch in range(epoch_start, epochs + 1):
			model.train()

			total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
			for label, euler, origin, rotate, _, _ in train_bar:
				origin, rotate = origin.cuda(), rotate.cuda()
				label, euler = label.cuda(), euler.cuda()

				pred_labels, pred_eulers = net(origin, rotate)
				loss_m = nn.MSELoss()
				loss_task = nn.CrossEntropyLoss()
				loss = loss_task(pred_labels, label) + w * loss_m(pred_eulers, euler)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_num += train_loader.batch_size
				total_loss += loss.item() * train_loader.batch_size
				train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, epochs, self.optimizer.param_groups[0]['lr'], total_loss / total_num))

			results['train_loss'].append(total_loss / total_num)

			model.eval()
			with torch.no_grad():
				test_bar = tqdm(self.test_loader)
				for label, euler, origin, rotate, _, _ in test_bar:
					origin, rotate = origin.cuda(), rotate.cuda()
					label, euler = label.cuda(), euler.cuda()

					pred_labels, _ = net(origin, rotate)

					total_num += origin.size(0)
					_, predicted = torch.max(pred_labels, dim=1)
					total_top1 += (predicted == label).float().sum().item()
					test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, epochs, total_top1 / total_num * 100))

			results['test_acc@1'].append(total_top1 / total_num * 100)

			self.scheduler.step()

			data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
			data_frame.to_csv(self.log_dir + '/log.csv', index_label='epoch')

			torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.log_dir + '/model_last.pth')

