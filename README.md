# Extreme Lightweight Portrait Segmentation



## Requirements

- python 3
- pytorch >= 0.4.1
- torchvision==0.2.1
- opencv-python==3.4.2.17
- numpy
- tensorflow >=1.13.0
- visdom


## Model
ExtremeC3Net ([paper](https://arxiv.org/abs/1908.03093))

Hyojin Park, Lars Lowe Sjösund, YoungJoon Yoo, Jihwan Bang, Nojun Kwak.
 
"ExtremeC3Net: Extreme Lightweight Portrait Segmentation Networks using Advanced C3-modules"

- config file : extremeC3Net.json
- Param : 0.038 M
- Flop : 0.128 G
- IoU : 94.98

SINet ([paper](https://arxiv.org/abs/1911.09099)) Accepted in WACV2020 

Hyojin Park, Lars Lowe Sjösund, YoungJoon Yoo, Nicolas Monet, Jihwan Bang, Nojun Kwak

SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder

- config file : SINet.json
- Param : 0.087 M
- Flop : 0.064 G
- IoU : 95.2 
## Run example



- Preparing dataset

Download datasets 
if you use audgmented dataset, fix the code in dataloader.py in line 20 depending on location of augmented dataset.
Also, please make different pickle file for Augmented dataset and baseline dataset.

- Train

1 . ExtremeC3Net
   
```shell
python main.py --c ExtremeC3Net.json
```
2 . SINet 
   
```shell
python main.py --c SINet.json
```
 


## Additonal Dataset

We make augmented dataset from Baidu fashion dataset.

The original Baidu dataset link is [here](http://www.cbsr.ia.ac.cn/users/ynyu/dataset/)

EG1800 dataset link what I used in [here](https://drive.google.com/file/d/18xM3jU2dSp1DiDqEM6PVXattNMZvsX4z/view?usp=sharing) 

Our augmented dataset is [here](https://drive.google.com/file/d/1zkh7gAhWwoX1nR5GzTzBziG8tgTKtr73/view?usp=sharing). 
We use all train and val dataset for training segmentation model. 

## CityScape
If you want SINet code for cityscapes dataset, please go to this [link](https://github.com/clovaai/c3_sinet).


## Citation
If our works is useful to you, please add two papers.
```shell
@article{park2019extremec3net,
  title={ExtremeC3Net: Extreme Lightweight Portrait Segmentation Networks using Advanced C3-modules},
  author={Park, Hyojin and Sj{\"o}sund, Lars Lowe and Yoo, YoungJoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:1908.03093},
  year={2019}
}

@article{park2019sinet,
  title={SINet: Extreme Lightweight Portrait Segmentation Networks with Spatial Squeeze Modules and Information Blocking Decoder},
  author={Park, Hyojin and Sj{\"o}sund, Lars Lowe and Monet, Nicolas and Yoo, YoungJoon and Kwak, Nojun},
  journal={arXiv preprint arXiv:1911.09099},
  year={2019}
}

```

## Acknowledge
We are grateful to [Clova AI, NAVER](https://github.com/clovaai) with valuable discussions.

I also appreciate my co-authors Lars Lowe Sjösund and YoungJoon Yoo from  [Clova AI, NAVER](https://clova.ai/en/research/research-areas.html),
Nicolas Monet from [NAVER LABS Europe](https://europe.naverlabs.com/)
and Jihwan Bang from [Search Solutions, Inc](https://www.searchsolutions.co.kr/)


## License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
