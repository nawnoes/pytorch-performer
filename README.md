# performer
Simple implementaion of pytorch Performer.
## Model
 Performer approximates kernel using random feature map. 
The kernel expects to replace Transformer's Dot-Product Self attention.
![](https://1.bp.blogspot.com/-pQ8s4X2qXjI/X5Ib6nLtxWI/AAAAAAAAGtI/C7dmMqV3Gu0NGYtmi5Gqjkr_Pqun5T2MwCLcBGAsYHQ/s1428/image10.jpg) 
### Softmax Kernel
`kernel_transformation=softmax_kernel_transformation.`
### Relu Kernel
`kernel_transformation=relu_kernel_transformation`
## Test &  Example
### Language Model
#### Pertrain
pretrain masked language model. 
- Pretrain file: `/example/train_mlm.py`.
- Config file: `/example/config.json`

##### Usage
① prepare dataset and vocab you want to train  
② check configuration in config.json  
③ run `/example/train_mlm.py`  

#### Finetuing


## TODO
- [ ] Performer performance test
- [ ] Write test example
- [ ] apply to language model
- [ ] evaluate language model
 

 
# References
- [Performer's Fast Attention (FAVOR+) Module](https://github.com/google-research/google-research/tree/master/performer/fast_attention)
- [Google Performer Blog](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)
- [teddykoker Performer](https://github.com/teddykoker/performer/blob/main/performer.py)
