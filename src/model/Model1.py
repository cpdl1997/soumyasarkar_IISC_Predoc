import torch
import torch.nn as nn

class Model1(torch.nn.Module):

    def __init__(self, _in_features, _out_features, _learning_rate=0.01):
        super().__init__()
        self._in_features = _in_features
        self._out_features = _out_features
        self._learning_rate = _learning_rate
        self.lin1 = nn.Linear(in_features = _in_features, out_features = 60)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features = 60, out_features = 30)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features = 30, out_features = _out_features)
        # self.act3 = self.softmax()
        self.act4 = nn.ReLU()
        
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        # result_tensor = torch.empty((0,1))
        # for i in x:
        #     print(i/10000)
        #     e_i = torch.exp(i/1e6) 
        #     print(e_i)
        #     res = e_i/torch.sum(e_i)
        #     print(res)
        #     result_tensor = torch.concat([result_tensor, res])
        #     print("result_tensor = ", result_tensor)
        # return result_tensor
        e_x = torch.exp(x/1e6)
        return e_x/torch.sum(e_x)

    
    def forward(self, train):
        x = self.act1(self.lin1(train))
        x = self.act2(self.lin2(x))
        x1 = self.lin3(x)[:,0:self._out_features-1]
        x1 = self.softmax(x1)
        x2 = self.act4(self.lin3(x)[:,self._out_features-1:self._out_features])
        output = (x1,x2)
        return output
