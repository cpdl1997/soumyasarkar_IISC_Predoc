import torch

ls = [1.3839e-03, 4.9662e-14, 2.3133e-20, 3.6332e-18, 8.8190e-01, 2.7283e-10,
         3.4796e-20, 4.2130e-08, 6.4545e-21, 8.0738e-02, 1.5078e-10, 8.0009e-18,
         5.4301e-04, 8.2980e-19, 1.7118e-11, 5.7588e-03, 2.9680e-02, 3.4262e-13,
         4.4492e-17, 5.2224e-13, 1.2594e-13, 3.8819e-13]

tensor_ls = torch.tensor(ls)

ls2 = [1.3839e-02, 4.9662e-10, 2.3133e-20, 3.6332e-18, 8.8190e-01, 2.7283e-1,
         3.4796e-20, 4.2130e-08, 6.4545e-21, 8.0738e-01, 1.5078e-10, 8.0009e-1,
         5.4301e-04, 8.2980e-1, 1.7118e-11, 5.7588e-05, 2.9680e-02, 3.4262e-13,
         4.4492e-17, 5.2224e-13, 1.2594e-13, 3.8819e-10]

sum = 0
for i in ls:
    sum+=i

tensor_ls2 = torch.tensor(ls2)

tensor_cmb = torch.cat((tensor_ls, tensor_ls2)).reshape(2,-1)
print(tensor_cmb)

topk_values, linear_indices = tensor_cmb.flatten().topk(4)
topk_indices = linear_indices % tensor_ls.shape[-1]


for i,j in enumerate(tensor_ls):
    if i in topk_indices:
        print("Yay!")
    else:
        print("Nooo")
    print(i)
    print(j)