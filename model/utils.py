import numpy as np
import torch

def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """
    
    # START CODE HERE
    # Get i from dimension span k
    i = k//2
    # Calculate the angles using pos, i and d
    angles = pos / np.power(10000, 2*i/d)
    # END CODE HERE
    
    return angles

def positional_encoding(positions=100, d=512):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded 
        d (int) -- Encoding size 
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # START CODE HERE
    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.array([[i] for i in range(positions)]),
                            np.array([[i for i in range(d)]]),
                            d)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:,0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:,1::2])
    # END CODE HERE
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    # output shape is [1,positions,d]
    return torch.from_numpy(pos_encoding)

pos_encoding = positional_encoding(100,2048).permute(0,2,1)
pos_encoding = pos_encoding.repeat(32,1,1)
# here pos_encoding is [N,F,T]
pos_encoding = pos_encoding.cuda()

def get_attn_local_mask(local_size=5, T=50, non_uniform=False):
    mask = torch.ones((T,T),dtype=torch.bool)
    if not non_uniform:
        for i in range(T):
            st, ed = i-local_size+1, i+local_size
            for j in range(st,ed):
                if j < 0 or j >= T:
                    continue
                mask[i,j] = False
    else:
        for i in range(T):
            local = min(i, T-1-i)
            st, ed = i-local, i+local+1
            for j in range(st,ed):
                mask[i,j] = False
    return mask.cuda()



if __name__ == '__main__':
    # prepare for MHA
    mask = get_attn_local_mask(3, 10, True)
    for row in mask:
        print(row)
