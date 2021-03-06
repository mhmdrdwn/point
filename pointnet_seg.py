import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=128)
        
        self.fc1 = nn.Conv1d(3,64,1)
        self.fc2 = nn.Conv1d(64,128,1) 
        self.fc3 = nn.Conv1d(128,128,1)
        self.fc4 = nn.Conv1d(128,512,1)
        self.fc5 = nn.Conv1d(512,2048,1)

        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

   def forward(self, input):
        n_pts = input.size()[2]
        matrix3x3 = self.input_transform(input)
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        outs = []
        
        out1 = F.relu(self.bn1(self.fc1(xb)))
        outs.append(out1)
        out2 = F.relu(self.bn2(self.fc2(out1)))
        outs.append(out2)
        out3 = F.relu(self.bn3(self.fc3(out2)))
        outs.append(out3)
        matrix128x128 = self.feature_transform(out3)
        
        out4 = torch.bmm(torch.transpose(out3,1,2), matrix128x128).transpose(1,2) 
        outs.append(out4)
        out5 = F.relu(self.bn4(self.fc4(out4)))
        outs.append(out5)
       
        xb = self.bn5(self.fc5(out5))
        
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        out6 = nn.Flatten(1)(xb).repeat(n_pts,1,1).transpose(0,2).transpose(0,1)#.repeat(1, 1, n_pts)
        outs.append(out6)
        
        
        return outs, matrix3x3, matrix128x128


class PointNetSeg(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.transform = Transform()

        self.fc1 = nn.Conv1d(3008,256,1) 
        self.fc2 = nn.Conv1d(256,256,1) 
        self.fc3 = nn.Conv1d(256,128,1) 
        self.fc4 = nn.Conv1d(128,num_classes,1) 
        

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(num_classes)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        

    def forward(self, input):
        inputs, matrix3x3, matrix128x128 = self.transform(input)
        
        stack = torch.cat(inputs,1)
        
        xb = F.relu(self.bn1(self.fc1(stack)))
       
        xb = F.relu(self.bn2(self.fc2(xb)))
    
        xb = F.relu(self.bn3(self.fc3(xb)))
        
        output = F.relu(self.bn4(self.fc4(xb)))
        
        return self.logsoftmax(output), matrix3x3, matrix128x128
    
    
def pointnetloss(outputs, labels, m3x3, m128x128, alpha = 0.0001):
    #print(outputs.shape, labels.shape)
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id128x128 = torch.eye(128, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id128x128=id128x128.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff128x128 = id128x128-torch.bmm(m128x128,m128x128.transpose(1,2))
    loss = criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff128x128)) / float(bs)
    return loss
    
#ref: https://github.com/nikitakaraevv/pointnet/
