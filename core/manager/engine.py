import os
import torch


class MTNetEngine():
    def __init__(self,
                 model,
                 optimizer,
                 config):
        self.config = config
        self.vos_model = model
        self.optimizer = optimizer
        self.save_path = config['save_path']
        self.exp = config['exp']
        self.save_path = os.path.join(self.exp, self.save_path)

        self.process_pretrained_model()

    def process_pretrained_model(self):
        if self.config['pretrain_model_path'] != None:
            self.load_network(self.config['pretrain_model_path'])

    def save_model(self, it, is_best=False, best_iou=None):
        if self.save_path is None:
            print("Saving has been disabled.")
            return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint = {
            'it': it,
            'network': self.vos_model.state_dict()}
        if is_best:
            best_iou = format(best_iou, '.4f')
            checkpoint_path = os.path.join(self.save_path, f'best_{str(it)}_{str(best_iou)}.pth')
            checkpoint_path_all = os.path.join(self.save_path, "best.pth")
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, checkpoint_path_all)
        else:
            checkpoint_path = os.path.join(self.save_path, f"{it}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    def save_checkpoint(self, it, optimizer=None, scaler=None):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = os.path.join(self.save_path, f"{it}_checkpoint.pth")
        checkpoint = {
            'it': it,
            'network': self.vos_model.state_dict(),
            'optimizer': optimizer.state_dict()}
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path, scaler=None):
        # This method loads everything and should be used to resume training
        self.local_rank = 0
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        if scaler is not None and 'scaler' in checkpoint.keys():
            scaler.load_state_dict(checkpoint['scaler'])
        self.vos_model.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)

        print('Model loaded.')
        del checkpoint
        return it

    def load_network(self, path):
        # This method only load model parameter and should be used to fine-tuning
        checkpoint = torch.load(path, map_location="cpu")
        new_param = checkpoint['network']
        # from IPython import embed
        # embed()
        # new_checkpoint = {'network':checkpoint['network']}
        try:
            self.vos_model.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            self.vos_model.load_state_dict(new_param)
        print('Network weight loaded:', path)
        del checkpoint
