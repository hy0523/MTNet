import argparse
import os.path


def none_or_default(x, default):
    return x if x is not None else default


class Configuration():
    def get_parser(self, unknown_arg_ok=False):
        parser = argparse.ArgumentParser(description='MTNet')
        # Environment
        parser.add_argument('--manual_seed', default=2023)
        parser.add_argument('-seed', dest='seed', default=2023, type=int)
        parser.add_argument('--seed_deterministic', type=bool, default=False)
        # Backbone
        parser.add_argument('--encoder_name', default='convnext-tiny')
        # Training parameters
        parser.add_argument('--imsize', dest='imsize', default=512, type=int)
        parser.add_argument('--train_frames', type=int, default=3)
        parser.add_argument('--batch_size', dest='batch_size', type=int)
        parser.add_argument('--num_workers', dest='num_workers', default=16, type=int)  # 8
        parser.add_argument('--no_amp', action='store_true')
        parser.add_argument('--warm_up_steps', default=0)
        parser.add_argument('--save_checkpoint_interval', type=int)
        parser.add_argument('--stage', type=str, default='main_train',
                            choices=['pre_train', 'main_train'])
        parser.add_argument('--max_iter', type=int)
        parser.add_argument('--resume', dest='resume', action='store_true',
                            help=('whether to resume training an existing model'
                                  '(the one with name model_name will be used)'))
        parser.add_argument('--pretrain_model_path', type=str, default=None)
        parser.add_argument('--save_path', type=str)
        parser.add_argument('-lr', dest='lr', type=float)
        parser.add_argument('-min_lr', dest='min_lr', type=float)
        parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
        parser.add_argument('-momentum', dest='momentum', default=0.9, type=float)
        parser.add_argument('--adamw_weight_decay', dest='adamw_weight_decay', default=0.05,
                            type=float)
        parser.add_argument('--crop', dest='crop', action='store_true')
        parser.set_defaults(crop=False)
        parser.add_argument('--update_encoder', dest='update_encoder',
                            action='store_true',
                            help='used in sync with finetune_after.'
                                 ' no need to activate.')
        parser.set_defaults(update_encoder=True)
        # Resume
        parser.add_argument('--resume_weight', default=None, type=str)
        # Eval
        parser.add_argument('--eval', default=True)
        parser.add_argument('--eval_iter', type=int)
        # Saving
        parser.add_argument('--exp', type=str, default='exp')
        parser.add_argument('--model_name', type=str, default='MTNet')
        # Visualization and logging
        parser.add_argument('--print_every', dest='print_every', default=10,
                            type=int)
        parser.add_argument('--do_log', default=True)
        parser.add_argument('--report_interval', type=int, default=100)
        parser.add_argument('--save_im_interval', type=int, default=100)
        parser.add_argument('--log_text_interval', default=100, type=int)
        parser.add_argument('--log_image_interval', default=200, type=int)
        # Augmentation
        parser.add_argument('--augment', dest='augment', action='store_true')
        parser.set_defaults(augment=True)
        parser.add_argument('-rotation', dest='rotation', default=10, type=int)
        parser.add_argument('-translation', dest='translation', default=0.1,
                            type=float)
        parser.add_argument('-shear', dest='shear', default=0.1, type=float)
        parser.add_argument('-zoom', dest='zoom', default=0.7, type=float)
        # Distribution & GPU
        # parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--syncbn', type=bool, default=True)
        parser.add_argument('--device', default='cuda', help='device id(i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument('--use_gpu', type=bool, default=True)
        parser.add_argument('--distributed', default=True)
        # parser.add_argument('-gpu_id', dest='gpu_id', default=[0, 1], type=int)
        parser.add_argument('-ngpus', dest='ngpus', default=4, type=int)
        parser.add_argument('-model_name', dest='model_name', default='model')
        parser.add_argument('-log_file', dest='log_file', default='train.log')
        # Optimizer
        parser.add_argument('--warmup', default=False)
        parser.add_argument('--power', default=0.9)  # 0 means no decay
        parser.add_argument('--index_split', default=-1)  # index for determining the parms group wiht 10x learning rate
        # Testing
        parser.add_argument('--clip_length', type=int, default=3)
        parser.add_argument('-eval_split', dest='eval_split', default='test')
        parser.add_argument('-mask_th', dest='mask_th', default=0.5, type=float)
        parser.add_argument('-max_dets', dest='max_dets', default=100, type=int)
        parser.add_argument('-min_size', dest='min_size', default=0.001,
                            type=float)
        parser.add_argument('--display', dest='display', action='store_true')
        parser.add_argument('--no_display_text', dest='no_display_text',
                            action='store_true')
        parser.add_argument('--task_name', default='UVOS')
        parser.add_argument('--test_dataset', default='DAVIS16')
        parser.add_argument('--test_length', default=9)  # clip length in MTT
        parser.add_argument('--test_model', default='./saves/s2_mtnet.pth')
        parser.add_argument('--root', default='../data')
        parser.add_argument('--test_size', default=(512, 512))
        parser.add_argument('--output_dir', default='output')
        parser.add_argument('--benchmark', default=True)
        # Visualize
        parser.add_argument('--vis_dataset', type=list, default=['DAVIS16'])

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())
        self.args['amp'] = not self.args['no_amp']
        # Stage
        if self.args['stage'] == 'pre_train':
            self.args['save_path'] = os.path.join(self.args['model_name'], self.args['stage'])
            self.args['lr'] = none_or_default(self.args['lr'], 1e-4)
            self.args['batch_size'] = none_or_default(self.args['batch_size'], 4)
            self.args['max_iter'] = none_or_default(self.args['max_iter'], 80000)
            self.args['save_checkpoint_interval'] = none_or_default(self.args['save_checkpoint_interval'], 1000)
            self.args['eval_iter'] = none_or_default(self.args['eval_iter'], 4000)
        elif self.args['stage'] == 'main_train':
            self.args['save_path'] = os.path.join(self.args['model_name'],
                                                  self.args['stage'])
            self.args['lr'] = none_or_default(self.args['lr'], 1e-4)
            self.args['batch_size'] = none_or_default(self.args['batch_size'], 2)
            self.args['max_iter'] = none_or_default(self.args['max_iter'], 10000)
            self.args['save_checkpoint_interval'] = none_or_default(self.args['save_checkpoint_interval'],
                                                                    1000)
            self.args['eval_iter'] = none_or_default(self.args['eval_iter'], 1000)
        else:
            raise NotImplementedError
        # Encoder
        self.args['in_channels'] = [96, 192, 384, 768]
        self.args['out_channels'] = [96, 96, 96, 96]
        self.args['proj_dim'] = 96
        self.args['adamw_weight_decay'] = 0.05
        self.args['warm_up_steps'] = 1500
        self.args['power'] = 1.0
        # MTNet
        # MTT
        self.args['head'] = 4
        self.args['mtt_layers'] = 4
        self.args['dropout'] = 0.
        self.args['zone_size'] = 2
        # CTD
        self.args['ctd_layers'] = 4

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
