# AEMFOD → Ultralytics YOLOv8 / YOLO11 迁移包

自定义结构全部集中在 **`aemfod.py`**。AEFPN 的连接路径由 YAML 表达，AWCF、ALF、残差细化块及其必要辅助层由该文件实现。

## 迁移配置

### 1 新增模块文件

将 `aemfod.py` 复制到：

```text
ultralytics/nn/modules/aemfod.py
```

### 2 导出模块

在 `ultralytics/nn/modules/__init__.py` 末尾增加：

```python
from .aemfod import AWCF2, AWCF3, AWCF4, ALF1, ALF2, AEFPNBlock

__all__ += ("AWCF2", "AWCF3", "AWCF4", "ALF1", "ALF2", "AEFPNBlock")
```

### 3 导入到模型解析器

在 `ultralytics/nn/tasks.py` 的 imports 中增加：

```python
from ultralytics.nn.modules.aemfod import AWCF2, AWCF3, AWCF4, ALF1, ALF2, AEFPNBlock
```

### 4 在 `parse_model()` 中解析通道和输入列表

插入以下分支：

```python
elif m in {AWCF2, AWCF3, AWCF4, ALF1, ALF2}:
    if n != 1 or not isinstance(f, list):
        raise ValueError("AEMFOD fusion layers require one repeat and a list of inputs")
    c2 = make_divisible(min(args[0], max_channels) * width, 8)
    args = [c2, *args[1:], [ch[x] for x in f]]
elif m is AEFPNBlock:
    c1 = ch[f]
    c2 = make_divisible(min(args[0], max_channels) * width, 8)
    args = [c1, c2, n, *args[1:]]
    n = 1
```

### 5 放置配置

将本包 `configs/` 下的六份 YAML 复制到：

```text
ultralytics/cfg/models/aemfod/
```

也可以从任意位置以完整 YAML 路径创建模型，不必复制配置。
