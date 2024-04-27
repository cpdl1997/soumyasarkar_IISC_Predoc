import torch


#Based on https://stackoverflow.com/questions/58984043/how-to-retrieve-topk-values-across-multiple-rows-along-with-their-respective-ind
def topk_genres(output_genres, k=4, prob_sum_thresold=0.95):
    for row in output_genres: #We have to do it element-wise. Using flatten() will consider topk over all m*n elements
        _, linear_indices = row.topk(k)
        topk_indices = linear_indices % row.shape[-1]
        sum=0
        for i, val in enumerate(row):
            if i in topk_indices and sum<=prob_sum_thresold: #We consider upto k top elements or till probability sum exceeds a threshold, whichever comes earlier
                sum+=val.data
                with torch.no_grad():
                    row[i]=torch.tensor(1) #If in the topk of that row, add as possible genre
            else:
                with torch.no_grad():
                    row[i]=torch.tensor(0) #if not in the topk, not a possible genre
    return output_genres