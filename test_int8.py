import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from DnCNN_ReRAM import DnCNN
from utils import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/DnCNN-S-15", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--qn_on", type=bool, default=0, help='int quant')
parser.add_argument("--fp_on", type=int, default=0, help='int quant')
parser.add_argument("--weight_bit", type=int, default=8, help='int quant')
parser.add_argument("--output_bit", type=int, default=8, help='int quant')
parser.add_argument("--isint", type=bool, default=0, help='int quant')
parser.add_argument("--quant_type", type=str, default='group', help='int quant')
parser.add_argument("--left_shift_bit", type=int, default=3, help='int quant')

opt = parser.parse_args()
'''
python test_int8.py \
  --num_of_layers 17 \
  --logdir "logs/DnCNN-S-15-ReRAM" \
  --test_data Set12 \
  --test_noiseL 15 \
  --qn_on 1 \
  --weight_bit 8 \
  --output_bit 8 
'''

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers, qn_on=opt.qn_on, fp_on=opt.fp_on,
                weight_bit=opt.weight_bit, output_bit=opt.output_bit, isint=opt.isint, quant_type=opt.quant_type, left_shift_bit=opt.left_shift_bit)

    device_ids = list(range(torch.cuda.device_count()))
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
