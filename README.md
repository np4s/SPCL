# SPCL in PyTorch


This is the official PyTorch implementation of the SPCL module proposed in ''*Leveraging Self-Paced Curriculum Learning for Enhanced Modality Balance in Multimodal Conversational Emotion Recognition*'', which is a flexible plug-in module used for addressing imbalance learning across modalities in Multimodal
Conversational Emotion Recognition task.

**Paper Title: "Leveraging Self-Paced Curriculum Learning for Enhanced Modality Balance in Multimodal Conversational Emotion Recognition"**

**Authors: Phuong-Anh Nguyen, The-Son Le, Duc-Trong Le, Cam-Van Thi Nguyen\***

## Motivation
In multimodal tasks, due to *Property Discrepancy* and *Quality Discrepancy* across modalities i.e. modality imbalance, the effectiveness of multimodal integration is often limited, leading to **suboptimal modality intergration and reduction in overall performance**. We conducted an experiment comparing the performance of tri-modal model with their bi-modal versions. From the results, it can be seen that the bi-modal models occansionally outperform their tri-modal counterparts, depicted by the negative values of Gain(%).
<div  align="center">    
<img src="assets/motivation.PNG" width = "65%" />
</div>

## Method
Framework for SPCL method consists two sub-modules:
1. Modality Prediction, which leverages available Emotion Recognition model to generate uni-modal and cross-modal predictions;
2. Self-paced Curriculum Learning-based (SPCL), our main proposed module, which designs learning curricula to address imbalance learning across modalities throughout training phase via 2 components:
    - Difficulty Measurer for determining the difficulty of samples in the dataset
    - Learning Scheduler for controlling the learning pace    
<div  align="center">    
<img src="assets/model.PNG" width = "65%" />
</div>


## Dependencies
+ python 3.11
+ Other required packages are specified in `requirements.txt`

## Usage

### Data

Processed IEMOCAP and MELD datasets are provided in the directory ```data/```.

### Core Code

The core abstract code part is as following:
```python
    ---in training step ---
    
    # Calculate the SPCL loss w.r.t current batch.
    nll, ratio, take_samp, uni_nll = model.get_loss(data)    

    # Calculate gradient w.r.t the SPCL loss.
    loss = nll.item()
    _loss += loss
    for m in modalities:
        loss_m[m] += uni_nll[m].item()
    nll.backward()
    
    # Optimize the parameters.
    optimizer.step()
    
    ---continue for next training step---

    # Update curriculumn threshold after every epoch.
    if args.use_cl:
        model.increase_threshold()

```

## Reproduction of Experiment Results

### Hyperparameter Settings

$\varepsilon$ and $\alpha$ from the paper, i.e. `cl_threshold` and `cl_growth` respectively, are set up during experiment as follow:

#### IEMOCAP

|      | Modal | cl_growth | cl_threshold |
|-------------|--------|------------|---------------|
| **BIDDIN**      | atv    | 1.85       | 0.7           |
|             | at     | 1.85       | 1             |
|             | tv     | 1.85       | 0.7           |
|             | av     | 1.85       | 0.4           |
| **DialogueGCN** | atv    | 1.25       | 0.7           |
|             | at     | 1.25       | 0.65          |
|             | tv     | 1.2        | 0.7           |
|             | av     | 1.25       | 0.8           |
| **MM-DFN**      | atv    | 1.2        | 0.4           |
|             | at     | 1.1        | 0.35          |
|             | tv     | 1.25       | 0.4           |
|             | av     | 1.1        | 0.35          |
| **MMGCN**       | atv    | 1.1        | 0.4           |
|             | at     | 1.4        | 1.3           |
|             | tv     | 1.4        | 1.3           |
|             | av     | 1.4        | 1.3           |


#### MELD

|            | Modal  | cl_growth  | cl_threshold  |
|------------|--------|------------|---------------|
| **BIDDIN**     | atv    | 1.05       | 0.75          |
|            | at     | 1.05       | 1             |
|            | tv     | 1.05       | 0.8           |
|            | av     | 1.05       | 0.75          |
| **DialogueGCN**| atv    | 1.25       | 0.7           |
|            | at     | 1.25       | 0.7           |
|            | tv     | 1.25       | 0.7           |
|            | av     | 1.25       | 0.7           |
| **MM-DFN**     | atv    | 1.6        | 0.2           |
|            | at     | 1.6        | 0.2           |
|            | tv     | 1.8        | 0.2           |
|            | av     | 1.6        | 0.2           |
| **MMGCN**      | atv    | 1.25       | 0.4           |
|            | at     | 1.7        | 0.4           |
|            | tv     | 1.6        | 0.4           |
|            | av     | 1.6        | 0.4           |


### Training Scripts

Example scripts for training with backbone-specific hyperparameters are provided in the directory ```scripts/```.

## Citation

If you find this work useful, please consider citing it.

<!-- <pre><code>
@inproceedings{Peng2022Balanced,
  title	= {Balanced Multimodal Learning via On-the-fly Gradient Modulation},
  author = {Peng, Xiaokang and Wei, Yake and Deng, Andong and Wang, Dong and Hu, Di},
  booktitle	= {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year	= {2022}
}
</code></pre> -->
<!-- 
## Acknowledgement

## License

## Contact us
-->