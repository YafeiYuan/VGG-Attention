import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        # return out,attention
        return out
    

# 加载vgg16预训练后模型
vgg16 = torchvision.models.vgg16(pretrained=True)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = []

        # 第一个卷积部分 加载vgg16模型中的第一个block
        # 112, 112, 64
        vgg.append(vgg16.features[0])
        vgg.append(vgg16.features[1])
        vgg.append(vgg16.features[2])
        vgg.append(vgg16.features[3])
        vgg.append(vgg16.features[4])

        # 第二个卷积部 加载vgg16模型中的第二个block
        # 56, 56, 128
        vgg.append(vgg16.features[5])
        vgg.append(vgg16.features[6])
        vgg.append(vgg16.features[7])
        vgg.append(vgg16.features[8])
        vgg.append(vgg16.features[9])

        # 第三个卷积部 加载vgg16模型中的第三个block
        # 28, 28, 256
        vgg.append(vgg16.features[10])
        vgg.append(vgg16.features[11])
        vgg.append(vgg16.features[12])
        vgg.append(vgg16.features[13])
        vgg.append(vgg16.features[14])
        vgg.append(vgg16.features[15])
        vgg.append(vgg16.features[16])

        # 第四个卷积部 加载vgg16模型中的第四个block
        # 14, 14, 512
        vgg.append(vgg16.features[17])
        vgg.append(vgg16.features[18])
        vgg.append(vgg16.features[19])
        vgg.append(vgg16.features[20])
        vgg.append(vgg16.features[21])
        vgg.append(vgg16.features[22])
        vgg.append(vgg16.features[23])

        # 第五个卷积部 加载vgg16模型中的第五个block
        # 7, 7, 512
        vgg.append(vgg16.features[24])
        vgg.append(vgg16.features[25])
        vgg.append(SelfAttention(512))  # 加入selfattention
        vgg.append(vgg16.features[26])
        vgg.append(vgg16.features[27])
        vgg.append(SelfAttention(512))  # 加入selfattention
        vgg.append(vgg16.features[28])
        vgg.append(vgg16.features[29])
        vgg.append(vgg16.features[30])
        vgg.append(SelfAttention(512))  # 加入selfattention

        # 将每一个模块按照他们的顺序送入到nn.Sequential中,输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化
        self.main = nn.Sequential(*vgg)

        self.avgpool = vgg16.avgpool

        # 全连接层 加入vgg16模型中的全连接层
        classfication = []
        # in_features四维张量变成二维[batch_size,channels,width,height]变成[batch_size,channels*width*height]
        classfication.append(vgg16.classifier[0])
        classfication.append(vgg16.classifier[1])
        classfication.append(vgg16.classifier[2])
        classfication.append(vgg16.classifier[3])
        classfication.append(vgg16.classifier[4])
        classfication.append(vgg16.classifier[5])
        # classfication.append(vgg16.classifier[6])
        classfication.append(nn.Linear(in_features=4096, out_features=6, bias=True))

        self.classfication = nn.Sequential(*classfication)

    def forward(self, x):
        feature = self.main(x)  # 输入张量x
        # print(feature.shape)
        feature = self.avgpool(feature)
        feature = feature.view(x.size(0), -1)  # reshape x变成[batch_size,channels*width*height]
        result = self.classfication(feature)
        return result
    
# 这种重写方式失去了 预训练部分 但是可以下载网络中vgg16预训练好的模型并加载进来也是可以的。
class VGG16(nn.Module):
    def __init__(self, nums, pretrained):
        super(VGG16, self).__init__()
        self.nums = nums
        vgg = []

        if pretrained: # 传入预训练后vgg16模型URL
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            vgg = model.load_state_dict(state_dict)

        # 第一个卷积部分
        # 112, 112, 64
        vgg.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第二个卷积部分
        # 56, 56, 128
        vgg.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第三个卷积部分
        # 28, 28, 256
        vgg.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第四个卷积部分
        # 14, 14, 512
        vgg.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第五个卷积部分
        # 7, 7, 512
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU(inplace=True))
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 将每一个模块按照他们的顺序送入到nn.Sequential中,输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化
        self.main = nn.Sequential(*vgg)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        # 全连接层
        classfication = []
        # in_features四维张量变成二维[batch_size,channels,width,height]变成[batch_size,channels*width*height]
        classfication.append(nn.Linear(in_features=512 * 7 * 7, out_features=4096))  # 输出4096个神经元，参数变成512*7*7*4096+bias(4096)个
        classfication.append(nn.ReLU(inplace=True))
        classfication.append(nn.Dropout(p=0.5))
        # classfication.append(nn.Linear(in_features=4096, out_features=4096))
        # classfication.append(nn.ReLU(inplace=True))
        # classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=1000))
        classfication.append(nn.ReLU(inplace=True))
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=1000, out_features=self.nums))

        self.classfication = nn.Sequential(*classfication)

    def forward(self, x):
        feature = self.main(x)  # 输入张量x
        # print(feature.shape)
        feature = self.avgpool(feature)
        feature = feature.view(x.size(0), -1)  # reshape x变成[batch_size,channels*width*height]
        result = self.classfication(feature)
        return result

