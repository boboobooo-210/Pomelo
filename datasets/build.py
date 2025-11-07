from utils import registry


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(cfg, default_args = default_args)

# 最终DATASETS.build返回的是obj_cls(cfg)，即注册数据集类（数据集配置yaml）
# obj_cls(cfg)初始化构造file_list.数据集对象的getItem返回【id,id,data】，如ShapeNet主要返回(N,1024,3)大小的点云数据
#
