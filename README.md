# AdaPLM-TabLS Framework

Code is adapted from [1]. Use the `--conversion` argument for performing the data conversion step. To characterize the augmented training data (returned from `train_summary_boosting` with `reuse` set to `True`), please refer to the documentation of [Data Maps](https://github.com/allenai/cartography) [2]. Other implementation details are described in the paper's Appendix D.

[1] H. Manikandan, Y. Jiang, and J. Z. Kolter, "Language models are weak learners," Advances in Neural Information Processing Systems, vol. 36, pp. 50 907–50 931, 2023.

[2] S. Swayamdipta et al., "Dataset cartography: Mapping and diagnosing datasets with training dynamics," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), B. Webber, T. Cohn, Y. He, and Y. Liu, Eds., Online: Association for Computational Linguistics, Nov. 2020, pp. 9275–9293.