import torch
import torchvision.datasets
from torch.utils.data import dataset
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))])

norm_tensor_to_image = transforms.Compose([
    transforms.Normalize((-1., -1., -1.),
                         (2., 2., 2.)),
    transforms.ToPILImage()])


def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=False,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=False,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fcl = nn.Sequential(
            nn.Linear(100, 256),
            nn.ELU()
        )

        self.conv = nn.Sequential(
           nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=6, padding=0),
           nn.ELU(),

           nn.UpsamplingBilinear2d(scale_factor=2),
           nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=0),
           nn.ELU(), 

           nn.UpsamplingBilinear2d(scale_factor=2),
           nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=0),
           nn.ELU(), 

           nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, padding=0),
           nn.Tanh()
        )

    def forward(self, input_z):
        return self.conv(self.fcl(torch.squeeze(input_z)).view(-1, 256, 1, 1))



class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=1024*4*4, out_features=1)
        )

    def forward(self, input):
        x = self.conv(input).view(-1, 1024*4*4)
        return self.output(x)


class WDLoss(nn.Module):
    def __init__(self):
        super(WDLoss, self).__init__()

    def forward(self, real, fake, gradient_z, lambda_gp):
        return real.mean() - fake.mean() - lambda_gp * ((gradient_z.norm(p=2, dim=1) - 1) ** 2).mean()


if __name__ == '__main__':
    # batch_size = 64
    # train_loader, valid_loader, test_loader = get_data_loader("svhn", batch_size)
    #
    # if torch.cuda.is_available():
    #     print("Using the GPU")
    #     device = torch.device("cuda")
    #     cuda_available = True
    #
    # else:
    #     print("WARNING: You are about to run on cpu, and this will likely run out \
    #     of memory. \n You can try setting batch_size=1 to reduce memory usage")
    #     device = torch.device("cpu")
    #     cuda_available = False
    #
    # critic = Critic()
    # generator = Generator()
    #
    # if cuda_available:
    #     critic, generator = critic.cuda(), generator.cuda()
    #
    # criterion = WDLoss()
    #
    # # setup optimizers
    # optim_d = optim.Adam(critic.parameters(), lr=3e-4, betas=(0.5, 0.999))
    # optim_g = optim.Adam(generator.parameters(), lr=3e-4, betas=(0.5, 0.999))
    #
    # lambda_gp = 10
    #
    # iterations_d = 5
    #
    # for epoch in range(25):
    #
    #     for i, (data, label) in enumerate(train_loader):
    #
    #         if data.shape[0] != batch_size:
    #             break
    #
    #         for p in critic.parameters():
    #             p.requires_grad = True
    #
    #         ###########################
    #         # (1) Update Critic network
    #         ###########################
    #
    #         # train with real
    #         real = data.cuda()
    #         d_real = critic(real)
    #
    #         # train with fake
    #         noise = torch.randn(batch_size, 100, 1, 1).cuda()
    #         fake = generator(noise)
    #         d_fake = critic(fake)
    #
    #         # train with gradient penalty
    #         a = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1).cuda()
    #         a = a.expand(batch_size, real.size(1), real.size(2), real.size(3))
    #         z = Variable(a * real + (1 - a) * fake, requires_grad=True).cuda()
    #
    #         optim_d.zero_grad()
    #
    #         z_out = critic(z)
    #         gradient_z = torch.autograd.grad(outputs=z_out, inputs=z, grad_outputs=torch.ones(z_out.size()).cuda(),
    #                                          retain_graph=True, create_graph=True)[0]
    #         loss = -criterion(d_real, d_fake, gradient_z, lambda_gp)
    #         loss.backward()
    #         optim_d.step()
    #
    #         if (i+1) % iterations_d == 0:
    #             ############################
    #             # (2) Update Generator network:
    #             ###########################
    #
    #             for p in critic.parameters():
    #                 p.requires_grad = False
    #
    #             noise = torch.randn(batch_size, 100, 1, 1).cuda()
    #             optim_g.zero_grad()
    #             fake = generator(noise)
    #             g_fake = -critic(fake).mean()
    #             g_fake.backward()
    #             optim_g.step()
    #             print(-g_fake)
    #
    #     fake = generator(torch.randn(25, 100, 1, 1).cuda()).data.cpu()
    #     fig = plt.figure(figsize=(10, 10))
    #     for i in range(0, 25):
    #         img = norm_tensor_to_image(fake[i, :, :, :])
    #         fig.add_subplot(5, 5, i+1)
    #         plt.imshow(img)
    #     plt.savefig('fake_sample_epoch_b%d.png' % epoch)
    #
    # torch.save(generator.state_dict(), "best_wgan_generator2.pt")

    device = torch.device('cpu')
    best_generator = Generator()
    best_generator.load_state_dict(torch.load('best_wgan_generator.pt'))
    best_generator.eval()

    # Qualitative
    # Question 1
    fig = plt.figure(figsize=(10, 10))
    sample_25 = best_generator(torch.randn((25, 100, 1, 1))).data.cpu()
    for i in range(25):
        img = norm_tensor_to_image(sample_25[i, :, :, :])
        fig.add_subplot(5, 5, i+1)
        plt.imshow(img)
    plt.show()

    # # Question 2
    # np.random.seed(2134)
    # z = np.random.normal(size=(1, 100, 1, 1)).astype(np.float32)
    # epsilon = 5
    # vec_eps = np.zeros(shape=(1, 100, 1, 1))
    # x = best_generator(torch.tensor(z)).data.cpu()
    # x = norm_tensor_to_image(x[0, :, :, :])
    # # plt.imshow(x)
    # # plt.show()

    # fig = plt.figure(figsize=(20, 20))
    # for i in range(100):
    #     vec_eps = np.zeros(shape=(1, 100, 1, 1)).astype(np.float32)
    #     vec_eps[0, i, 0, 0] = epsilon
    #     z_prime = z + vec_eps
    #     x = best_generator(torch.tensor(z_prime)).data.cpu()
    #     x = norm_tensor_to_image(x[0, :, :, :])
    #     fig.add_subplot(10, 10, i+1)
    #     plt.imshow(x)
    # plt.show()

    # interesting_list = [90, 75, 60, 69, 65]
    # fig = plt.figure(figsize=(2, 8))
    # fig.add_subplot(1, 6, 1)
    # plt.imshow(x)
    # plt.xlabel('ORIGINAL')
    # for i in range(len(interesting_list)):
    #     vec_eps = np.zeros(shape=(1, 100, 1, 1)).astype(np.float32)
    #     vec_eps[0, interesting_list[i], 0, 0] = epsilon
    #     z_prime = z + vec_eps
    #     x = best_generator(torch.tensor(z_prime)).data.cpu()
    #     x = norm_tensor_to_image(x[0, :, :, :])
    #     fig.add_subplot(1, 6, i+2)
    #     plt.imshow(x)
    # plt.show()

    # Question 3
    # (a)_

    # np.random.seed(217)
    # z_0 = np.random.normal(size=(1, 100, 1, 1)).astype(np.float32)
    # np.random.seed(37997)
    # z_1 = np.random.normal(size=(1, 100, 1, 1)).astype(np.float32)
    #
    # x_0 = best_generator(torch.tensor(z_0)).data.cpu()
    # x_0 = norm_tensor_to_image(x_0[0, :, :, :])
    # plt.imshow(x_0)
    # plt.show()
    #
    # x_1 = best_generator(torch.tensor(z_1)).data.cpu()
    # x_1 = norm_tensor_to_image(x_1[0, :, :, :])
    # plt.imshow(x_1)
    # plt.show()
    #
    # alpha = np.linspace(0, 1, 11).astype(np.float32).reshape((11, 1, 1, 1))
    # z_prime = torch.tensor(alpha * z_0 + (1. - alpha) * z_1)
    # samples = best_generator(z_prime).data.cpu()
    # fig = plt.figure(figsize=(2, 22))
    # for i in range(11):
    #     img = norm_tensor_to_image(samples[i, :, :, :])
    #     fig.add_subplot(1, 11, i+1)
    #     plt.imshow(img)
    # plt.show()
    #
    # # (b)
    # x_0 = best_generator(torch.tensor(z_0)).data.numpy()
    # x_1 = best_generator(torch.tensor(z_1)).data.numpy()
    # x_prime = torch.tensor(alpha * x_0 + (1. - alpha) * x_1)
    # fig = plt.figure(figsize=(2, 22))
    # for i in range(11):
    #     img = norm_tensor_to_image(x_prime[i, :, :, :])
    #     fig.add_subplot(1, 11, i+1)
    #     plt.imshow(img)
    # plt.show()

    # Quantitative
    # sample_1000 = best_generator(torch.randn((1000, 100, 1, 1))).data.cpu()
    # for i in range(1000):
    #     img = norm_tensor_to_image(sample_1000[i, :, :, :])
    #     plt.imshow(img)
    #     plt.savefig('C:/Users/Cedric/Dropbox/University/Doctorat/Cours/Apprentissage de Representations/Assignment 3/Practice/samples/sample_best_wgan%d.png' % i)






