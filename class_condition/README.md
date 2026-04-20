(1)每个class 生成50张图片，构造candidate集合，共8个集合（seed=0,1,2,3,...8）-> random search的评测集
(2)构造ranker的训练集，每个class生成128张图片并提取表征，然后train
(3)评测不同reward的random search和ranker的FK
(4)ablation

