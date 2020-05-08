'''
Copyright © 2020 IBM. This notebook and its source code are released under
the terms of the MIT License.

The PyTorch implementation of the Cycle Generative Adversarial Network (GAN)
model is based on the following paper and the GitHub repository.

Jun-Yan Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks," International Conference on Computer Vision, 2017.

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

BSD License

For pix2pix software
Copyright © 2016, Phillip Isola and Jun-Yan Zhu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
'''

from argparse import ArgumentParser
from itertools import chain
from numpy import maximum, tile, transpose, uint8
from os import environ, makedirs, system, walk
from os.path import isdir, join
from PIL import Image
from random import randint, random, uniform
from shutil import rmtree
from sys import stdout
from torch import cat, cuda, device, no_grad, save, tensor, unsqueeze
from torch.autograd import Variable
from torch.nn import Conv2d, ConvTranspose2d, Dropout, init, InstanceNorm2d,\
    L1Loss, LeakyReLU, Module, MSELoss, ReflectionPad2d, ReLU, Sequential,\
    Tanh
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Grayscale, Lambda, Normalize,\
    RandomCrop, Resize, ToTensor
from zipfile import ZipFile


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(file_name):
    return any(file_name.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(dir_name, max_dataset_size=float('inf')):
    images = []
    assert isdir(dir_name), '{} is not a valid directory'.format(dir_name)

    for root, _, files in sorted(walk(dir_name)):
        for f in files:
            if is_image_file(f):
                path = join(root, f)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def get_transform(
    params=None, grayscale=False, method=Image.BICUBIC, convert=True
):
    transform_list = []

    if grayscale:
        transform_list.append(Grayscale(1))

    osize = [286, 286]
    transform_list.append(Resize(osize, method))

    if params is None:
        transform_list.append(RandomCrop(256))
    else:
        transform_list.append(
            Lambda(lambda img: __crop(img, params['crop_pos'], 256))
        )

    if convert:
        transform_list += [ToTensor()]

        if grayscale:
            transform_list += [Normalize((0.5,), (0.5,))]
        else:
            transform_list += [Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return Compose(transform_list)


class HorseToZebraDataset(Dataset):
    def __init__(self, dataroot, phase, max_dataset_size=float('inf')):
        self.dir_A = join(dataroot, phase + 'A')
        self.dir_B = join(dataroot, phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_A = get_transform()
        self.transform_B = get_transform()

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, ix):
        A_path = self.A_paths[ix % self.A_size]
        ix_B = ix % self.B_size

        B_path = self.B_paths[ix_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


class ResnetBlock(Module):
    def __init__(self, dim, dropout):
        super(ResnetBlock, self).__init__()
        resnet_block = []

        resnet_block.append(ReflectionPad2d(1))
        resnet_block += [
            Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            InstanceNorm2d(dim),
            ReLU(True)
        ]

        if dropout:
            resnet_block.append(Dropout(.5))

        resnet_block.append(ReflectionPad2d(1))
        resnet_block += [
            Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            InstanceNorm2d(dim)
        ]

        self.resnet_block = Sequential(*resnet_block)

    def forward(self, x):
        return x + self.resnet_block(x)


class Generator(Module):
    def __init__(self, in_ch, out_ch, n_filters=64, n_blocks=6, dropout=False):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()

        model = [
            ReflectionPad2d(3),
            Conv2d(in_ch, n_filters, kernel_size=7, padding=0, bias=True),
            InstanceNorm2d(n_filters),
            ReLU(True)
        ]

        n_sampling = 2

        for i in range(n_sampling):
            mul = 2 ** i

            model += [
                Conv2d(
                    n_filters * mul, n_filters * mul * 2,
                    kernel_size=3, stride=2, padding=1, bias=True
                ),
                InstanceNorm2d(n_filters * mul * 2),
                ReLU(True)
            ]

        for i in range(n_blocks):
            model.append(
                ResnetBlock(
                    n_filters * (2 ** n_sampling), dropout=dropout
                )
            )

        for i in range(n_sampling):
            mul = 2 ** (n_sampling - i)
            model += [
                ConvTranspose2d(
                    n_filters * mul, int(n_filters * mul / 2),
                    kernel_size=3, stride=2, padding=1,
                    output_padding=1, bias=True
                ),
                InstanceNorm2d(int(n_filters * mul / 2)),
                ReLU(True)
            ]

        model.append(ReflectionPad2d(3))
        model.append(Conv2d(n_filters, out_ch, kernel_size=7, padding=0))
        model.append(Tanh())

        self.model = Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(Module):
    def __init__(self, in_ch, n_filters=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [
            Conv2d(in_ch, n_filters, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2, True)
        ]

        mul = 1
        mul_prev = 1

        for n in range(1, n_layers):
            mul_prev = mul
            mul = min(2 ** n, 8)

            layers += [
                Conv2d(
                    n_filters * mul_prev, n_filters * mul, kernel_size=4,
                    stride=2, padding=1, bias=True
                ),
                InstanceNorm2d(n_filters * mul),
                LeakyReLU(0.2, True)
            ]

        mul_prev = mul
        mul = min(2 ** n_layers, 8)

        layers += [
            Conv2d(
                n_filters * mul_prev, n_filters * mul, kernel_size=4,
                stride=1, padding=1, bias=True
            ),
            InstanceNorm2d(n_filters * mul),
            LeakyReLU(0.2, True)
        ]

        layers.append(
            Conv2d(n_filters * mul, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_net(net, init_gain=0.02):
    def init_func(n):
        class_name = n.__class__.__name__
        if hasattr(n, 'weight') and (
            class_name.find('Conv') != -1 or class_name.find('Linear') != -1
        ):
            init.normal_(n.weight.data, 0., init_gain)

    net.to(device('cuda'))
    return net.apply(init_func)


def get_scheduler(optimizer):
    def lambda_rule(epoch):
        lr_l = 1. - max(0, epoch - 99) / 101.
        return lr_l
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


class ImagePool():
    def __init__(self, max_size):
        self.max_size = max_size
        if self.max_size > 0:
            self.imgs_len = 0
            self.images = []

    def sample(self, images):
        if self.max_size == 0:
            return images

        sampled_imgs = []

        for img in images:
            img = unsqueeze(img.data, 0)

            if self.imgs_len < self.max_size:
                self.imgs_len += 1
                self.images.append(img)
                sampled_imgs.append(img)
            else:
                p = uniform(0, 1)
                if p > 0.5:
                    ix = randint(0, self.max_size - 1)
                    tmp_img = self.images[ix].clone()
                    self.images[ix] = img
                    sampled_imgs.append(tmp_img)
                else:
                    sampled_imgs.append(img)

        return cat(sampled_imgs, 0)


class GANLoss(Module):
    def __init__(self, real_label=1., fake_label=0.):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', tensor(real_label))
        self.register_buffer('fake_label', tensor(fake_label))
        self.loss = MSELoss()

    def get_target_tensor(self, pred, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(pred)

    def __call__(self, pred, target_is_real):
        target_tensor = self.get_target_tensor(pred, target_is_real)
        return self.loss(pred, target_tensor)


class CycleGAN():
    def __init__(
        self, in_ch=3, out_ch=3, dim=3, n_filters=64, n_blocks=6, n_layers=3,
        lr=.0002, beta1=.5, beta2=.999, lambda_A=10., lambda_B=10.,
        lambda_idt=.5, init_gain=.02, max_size=15, dropout=False,
        is_train=True, cuda=True
    ):
        self.is_train = is_train
        self.lambda_idt = lambda_idt
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device('cuda' if cuda else 'cpu')
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.dim = dim
        self.init_gain = init_gain
        self.max_size = max_size
        self.dropout = dropout
        self.optimizers = []
        self.model_list = []

        gen_A = Generator(
            self.in_ch, self.out_ch, self.n_filters, self.n_blocks,
            self.dropout
        )
        self.gen_AtoB = init_net(gen_A, self.init_gain)
        gen_B = Generator(
            self.in_ch, self.out_ch, self.n_filters, self.n_blocks,
            self.dropout
        )
        self.gen_BtoA = init_net(gen_B, self.init_gain)

        if self.is_train:
            self.model_names = ['gen_AtoB', 'gen_BtoA']

            d_A = Discriminator(self.dim, self.n_filters, self.n_layers)
            self.disc_A = init_net(d_A)
            d_B = Discriminator(self.dim, self.n_filters, self.n_layers)
            self.disc_B = init_net(d_B)

            if self.lambda_idt > 0.0:
                assert(self.in_ch == self.out_ch)

            self.fake_A_pool = ImagePool(self.max_size)
            self.fake_B_pool = ImagePool(self.max_size)
            self.crit_GAN = GANLoss().to(self.device)
            self.crit_cycle = L1Loss()
            self.crit_idt = L1Loss()
            self.opt_gen = Adam(
                chain(self.gen_AtoB.parameters(), self.gen_BtoA.parameters()),
                lr=self.lr, betas=(self.beta1, self.beta2)
            )
            self.opt_disc = Adam(
                chain(self.disc_A.parameters(), self.disc_B.parameters()),
                lr=self.lr, betas=(self.beta1, self.beta2)
            )
            self.optimizers.append(self.opt_gen)
            self.optimizers.append(self.opt_disc)
            self.schedulers = [
                get_scheduler(optimizer) for optimizer in self.optimizers
            ]

    def set_input(self, images):
        self.real_A = images['A'].to(self.device)
        self.real_B = images['B'].to(self.device)
        self.image_paths = images['A_paths']

    def forward(self):
        if self.is_train:
            self.gen_AtoB = (
                self.gen_AtoB if next(self.gen_AtoB.parameters()).is_cuda
                else self.gen_AtoB.to(self.device)
            )
            self.gen_BtoA = (
                self.gen_BtoA if next(self.gen_BtoA.parameters()).is_cuda
                else self.gen_BtoA.to(self.device)
            )
        else:
            assert(len(self.model_list) > 0)
            self.gen_AtoB = self.model_list[0]
            self.gen_BtoA = self.model_list[1]

        self.fake_B = self.gen_AtoB(self.real_A)
        self.reconstruct_A = self.gen_BtoA(self.fake_B)
        self.fake_A = self.gen_BtoA(self.real_B)
        self.reconstruct_B = self.gen_AtoB(self.fake_A)

    def backward_disc(self, disc, real, fake):
        pred_real = disc(real)
        loss_disc_real = self.crit_GAN(pred_real, True)

        pred_fake = disc(fake.detach())
        loss_disc_fake = self.crit_GAN(pred_fake, False)

        loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
        loss_disc.backward()
        return loss_disc

    def backward_disc_A(self):
        fake_B = self.fake_B_pool.sample(self.fake_B)
        self.loss_disc_A = self.backward_disc(self.disc_A, self.real_B, fake_B)

    def backward_disc_B(self):
        fake_A = self.fake_A_pool.sample(self.fake_A)
        self.loss_disc_B = self.backward_disc(self.disc_B, self.real_A, fake_A)

    def backward_gen(self):
        lambda_idt = self.lambda_idt
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        if lambda_idt > 0:
            self.idt_A = self.gen_AtoB(self.real_B)
            self.loss_idt_A = (
                self.crit_idt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            self.idt_B = self.gen_BtoA(self.real_A)
            self.loss_idt_B = (
                self.crit_idt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_gen_AtoB = self.crit_GAN(self.disc_A(self.fake_B), True)
        self.loss_gen_BtoA = self.crit_GAN(self.disc_B(self.fake_A), True)
        self.loss_cycle_A = (
            self.crit_cycle(self.reconstruct_A, self.real_A) * lambda_A
        )
        self.loss_cycle_B = (
            self.crit_cycle(self.reconstruct_B, self.real_B) * lambda_B
        )

        self.loss_gen = (
            self.loss_gen_AtoB + self.loss_gen_BtoA + self.loss_cycle_A +
            self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        )
        self.loss_gen.backward()

    def set_requires_grad(self, nn, requires_grad=False):
        if not isinstance(nn, list):
            nn = [nn]

        for net in nn:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_params(self):
        self.forward()

        self.set_requires_grad([self.disc_A, self.disc_B])
        self.opt_gen.zero_grad()
        self.backward_gen()
        self.opt_gen.step()

        self.set_requires_grad([self.disc_A, self.disc_B], True)
        self.opt_disc.zero_grad()
        self.backward_disc_A()
        self.backward_disc_B()
        self.opt_disc.step()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        return self.optimizers[0].param_groups[0]['lr']

    def save_networks(self, path):
        for name in self.model_names:
            if isinstance(name, str):
                filename = '{}_net.pth'.format(name)
                net = getattr(self, name)
                save(net.cpu().state_dict(), join(path, filename))

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def set_model_list(self, model_list):
        self.model_list = model_list

    def test(self):
        with no_grad():
            self.forward()
            img_dict = {}

            for i in range(2):
                image_tensor = (
                    self.fake_B.data if i % 2 == 0 else self.fake_A.data
                )
                image_numpy = image_tensor[0].cpu().float().numpy()

                if image_numpy.shape[0] == 1:
                    image_numpy = tile(image_numpy, (3, 1, 1))

                image_numpy = (
                    (transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
                )
                k = 'AtoB' if i % 2 == 0 else 'BtoA'
                img_dict[k] = image_numpy.astype(uint8)

            return img_dict


def train(cycle_gan, data_path, model_path, opt):
    with ZipFile(join(data_path, opt.train_images_file), 'r') as zfin:
        zfin.extractall(data_path)

    h2z_dataset = HorseToZebraDataset(
        join(data_path, 'zebra2horse'), 'train', opt.max_dataset_size
    )
    dataset = DataLoader(
        h2z_dataset, batch_size=opt.batch_size, shuffle=1, num_workers=4
    )
    total_iters = opt.epochs + 1

    for epoch in range(1, total_iters):
        for i, data in enumerate(dataset):
            cycle_gan.set_input(data)
            cycle_gan.optimize_params()

        if epoch == total_iters - 1:
            cycle_gan.save_networks(model_path)

        if opt.verbose == 1:
            print(
                'Epoch {}: learning rate = {:.7f}'.format(
                    epoch, cycle_gan.update_learning_rate()
                )
            )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='$DATA_DIR',
                        help='Directory with data')
    parser.add_argument('--result_dir', type=str, default='$RESULT_DIR',
                        help='Directory with results')
    parser.add_argument('--train_images_file', type=str,
                        default='zebra2horse.zip',
                        help='File name for train images')
    parser.add_argument('--in_ch', type=int, default=3,
                        help='Number of input image channels')
    parser.add_argument('--out_ch', type=int, default=3,
                        help='Number of output image channels')
    parser.add_argument('--dim', type=int, default=3,
                        help='Number of channels in the Convolution layer')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs with the initial learning rate')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='First Adam beta parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Second Adam beta parameter')
    parser.add_argument('--lambda_A', type=float, default=10.0,
                        help='Weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0,
                        help='Weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_idt', type=float, default=0.5,
                        help='Identity mapping. Setting lambda_identity other \
                              than 0 has an effect of scaling the weight of \
                              the identity mapping loss.')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='Scaling factor for normal network init.')
    parser.add_argument('--max_size', type=int, default=15,
                        help='Size of image buffer that stores \
                        previously generated images')
    parser.add_argument('--max_dataset_size', type=int, default=15,
                        help='Maximum number of images in the data set')
    parser.add_argument('--n_filters', type=int, default=64,
                        help='Number of filters of the last Conv layer of \
                        the Generator or the first Conv layer of the \
                        Discrimator')
    parser.add_argument('--n_blocks', type=int, default=6,
                        help='Number of Resnet blocks in the Generator')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of layers in the Discriminator')
    parser.add_argument('--dropout', action='store_false', default=False,
                        help='Use dropout for the Generator')
    parser.add_argument('--is_train', action='store_true', default=True,
                        help='Whether the model needs to be trained or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Size of the batch')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Show more details of training')

    FLAGS, unparsed = parser.parse_known_args()

    if (FLAGS.result_dir[0] == '$'):
        RESULT_DIR = environ[FLAGS.result_dir[1:]]
    else:
        RESULT_DIR = FLAGS.result_dir

    model_path = join(RESULT_DIR, 'model')

    try:
        makedirs(model_path)
    except FileExistsError:
        rmtree(model_path)
        makedirs(model_path)

    if (FLAGS.data_dir[0] == '$'):
        DATA_DIR = environ[FLAGS.data_dir[1:]]
    else:
        DATA_DIR = FLAGS.data_dir

    in_ch = FLAGS.in_ch
    out_ch = FLAGS.out_ch
    dim = FLAGS.dim
    n_filters = FLAGS.n_filters
    n_blocks = FLAGS.n_blocks
    n_layers = FLAGS.n_layers
    lr = FLAGS.lr
    beta1 = FLAGS.beta1
    beta2 = FLAGS.beta2
    lambda_A = FLAGS.lambda_A
    lambda_B = FLAGS.lambda_B
    lambda_idt = FLAGS.lambda_idt
    init_gain = FLAGS.init_gain
    max_size = FLAGS.max_size
    dropout = FLAGS.dropout
    is_train = FLAGS.is_train
    cuda = FLAGS.cuda

    cycle_gan = CycleGAN(
        in_ch=in_ch, out_ch=out_ch, dim=dim, n_filters=n_filters,
        n_blocks=n_blocks, n_layers=n_layers, lr=lr, beta1=beta1, beta2=beta2,
        lambda_A=lambda_A, lambda_B=lambda_B, lambda_idt=lambda_idt,
        init_gain=init_gain, max_size=max_size, dropout=dropout,
        is_train=is_train, cuda=cuda
    )

    train(cycle_gan, DATA_DIR, model_path, FLAGS)

    result_dir_env = environ['RESULT_DIR']
    system('(cd $RESULT_DIR/model)')
    stdout.flush()
