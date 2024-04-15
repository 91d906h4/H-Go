# H-Go

H-Go is a artificial intelligence for the game of Go.

## Introduction

In the realm of board games, Go has posed a profound challenge for AI systems due to its intricate rules and expansive search space. H-Go, with its approach, utilizes machine learning algorithms and straightforward decision-making strategies to approach human-level performance.

Powered by a blend of Convolutional Neural Networks (CNN), Self-Attention mechanisms, and Monte Carlo Tree Search (MCTS), H-Go demonstrates competent gameplay in the complex world of Go. Through ongoing self-improvement iterations and training on extensive datasets, H-Go continuously hones its gameplay tactics, adapting to diverse playing styles and evolving board situations.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install the required dependencies.

```sh
pip install -r requirements.txt
```

```sh
conda install --file requirements.txt
```

## Usage

1. Set Hyper-Parameters

Navigate to the `./src/hyperparameters.py` file and adjust the settings according to your requirements. This file contains essential configurations and parameters that influence the behavior and performance of the model. By modifying these settings, you can tailor the model's behavior to suit specific use cases and optimize its performance for desired outcomes.

2. Train the model

To begin, navigate to the `./src/main.ipynb` file and execute the notebook by selecting the "Run" option. Ensure that all required modules are correctly imported and that the parameters are appropriately configured. Upon execution, the notebook will initiate the training process for the model.

3. Play with model

Once the training is complete, you will have the opportunity to interact with the trained model directly within the notebook interface, located at the bottom of the file. Explore the functionalities and capabilities of the model to gain insights and assess its performance.

## TODO

- [ ] Illegal nodes (moves) pruning for HTS.
- [ ] Illegal moves checking and stone removal for game states.
- [ ] Value network overfitting.

## Contributing

We welcome pull requests from contributors. If you plan to make significant changes, kindly open an issue beforehand to initiate a discussion on the proposed modifications.

Ensure that appropriate tests are updated or added along with any changes made. 

Your attention to this matter is appreciated!

## License

H-Go is licensed under the [PROPRIETARY LICENSE](https://github.com/91d906h4/H-Go/blob/main/LICENSE), which prohibits any use, copying, modification, or distribution of the software without explicit written permission from the copyright holder.

## References

[1] D. Silver et al., "Mastering the game of Go with deep neural networks and tree search," Nature, vol. 529, pp. 484–489, Jan. 2016.<br />
[2] D. Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm," arXiv:1712.01815 [cs.AI], Dec. 2017. [Online]. Available: https://doi.org/10.48550/arXiv.1712.01815<br />
[3] Klein, "Alphagoについてのまとめ," kleinblog, 06 February 2021. [Online]. Available: https://kleinblog.net/alphago<br />
[4] T. Yamaoka, "AlphaGo Zeroの論文を読む その2(ネットワーク構成)," TadaoYamaokaの開発日記, 20 October 2017. [Online]. Available: https://tadaoyamaoka.hatenablog.com/entry/2017/10/20/221030<br />
[5] T. Yamaoka, "将棋でディープラーニングする その33(マルチタスク学習)," TadaoYamaokaの開発日記, 08 June 2017. [Online]. Available: https://tadaoyamaoka.hatenablog.com/entry/2017/06/08/000040<br />
[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," arXiv:1512.03385 [cs.CV], Dec. 2015. [Online]. Available: https://arxiv.org/pdf/1512.03385.pdf<br />
[7] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks," arXiv:1603.05027 [cs.CV], Mar. 2016. [Online]. Available: https://arxiv.org/abs/1603.05027<br />
[8] Deep Learning Training Challenges and Solutions: A Comprehensive Summary (OpenAI's ChatGPT, private communication, 10 April 2024).<br />
[9] Featurecat, "Fox Go Dataset," GitHub. [Online]. Available: https://github.com/featurecat/go-dataset<br />
[10] DOORS 編集部, "強化学習入門 Part3 － AlphaGoZeroでも重要な技術要素！ モンテカルロ木探索の入門 －," Brainpad, Nov. 1, 2020. [Online]. Available: https://www.brainpad.co.jp/doors/contents/01_tech_2018-04-05-163000/<br />
[11] leaderj1001, "Implementing Stand-Alone Self-Attention in Vision Models using Pytorch," GitHub. [Online]. Available: https://github.com/leaderj1001/Stand-Alone-Self-Attention/<br />
[12] P. Ramachandran et al., "Stand-Alone Self-Attention in Vision Models," arXiv:1906.05909 [cs.CV], Jun. 2019. [Online]. Available: https://arxiv.org/abs/1906.05909<br />
[13] Y. Yan, J. Kawahara, and G. Hamarneh, "Melanoma Recognition via Visual Attention," in Information Processing in Medical Imaging (IPMI 2019), pp. 793-804, May 2019.<br />
[14] J. Hui, "Monte Carlo Tree Search (MCTS) in AlphaGo Zero," Medium, May 20, 2018. [Online]. Available: https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a<br />
[15] J. Hui, "AlphaGo Zero — a game changer. (How it works?)" Medium, May 17, 2018. [Online]. Available: https://jonathan-hui.medium.com/alphago-zero-a-game-changer-14ef6e45eba5<br />
[16] M. Zhong, "蒙特卡洛树搜索（MCTS）代码详解【python】," CSDN, Mar. 23, 2019. [Online]. Available: https://blog.csdn.net/windowsyun/article/details/88770799<br />
[17] F. Opolka, "Single-Player Monte-Carlo Tree Search," GitHub, Jun. 25, 2021. [Online]. Available: https://github.com/FelixOpolka/Single-Player-MCTS/