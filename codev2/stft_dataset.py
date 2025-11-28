"""
Là dataloader cho training stage
"""
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Union

class STFTDataset(Dataset):
    """
    Đọc các gói dữ liệu tạo bởi preprocessing mới:
      {
        "inputs": [Tensor(2F, T), ...] hoặc Tensor(N, 2F, T)
        "targets": [Tensor(2F, T), ...] hoặc Tensor(N, 2F, T)
        "meta": [dict, ...]
        "frequencies": Tensor(F,)
        "times": Tensor(T,)
        "fs": int
        "seg_len": int
      }
    Trả về:
      - "input":  Tensor(1, 2F, T)
      - "target": Tensor(1, 2F, T)
      - "meta":   dict cho mẫu đó (nếu có)
      - "frequencies": Tensor(F,)
      - "times":  Tensor(T,)
    """

    def __init__(self, data_file: str):
        pkg: Dict[str, Any] = torch.load(data_file)

        self._inputs = pkg["inputs"]
        self._targets = pkg["targets"]
        self._meta = pkg.get("meta", None)
        self.frequencies = pkg.get("frequencies", None)
        self.times = pkg.get("times", None)
        self.fs = pkg.get("fs", None)
        self.seg_len = pkg.get("seg_len", None)

        # Chấp nhận cả list các tensor hoặc 1 tensor stack
        if isinstance(self._inputs, torch.Tensor):
            # kỳ vọng (N, 2F, T)
            assert self._inputs.dim() == 3, "inputs tensor phải có shape (N, 2F, T)"
            self._as_tensor_inputs = True
        else:
            # list[Tensor(2F, T)]
            self._as_tensor_inputs = False
        if isinstance(self._targets, torch.Tensor):
            assert self._targets.dim() == 3, "targets tensor phải có shape (N, 2F, T)"
            self._as_tensor_targets = True
        else:
            self._as_tensor_targets = False

        # Độ dài
        self._length = (self._inputs.shape[0] if self._as_tensor_inputs
                        else len(self._inputs))

    def __len__(self) -> int:
        return self._length

    def _get_sample(self, seq: Union[List[torch.Tensor], torch.Tensor], idx: int) -> torch.Tensor:
        x = seq[idx] if not isinstance(seq, torch.Tensor) else seq[idx]
        # kỳ vọng (2F, T)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, 2F, T)
        elif x.dim() == 3:
            # nếu đã (C, 2F, T) thì giữ nguyên, còn nếu là (2F, T, ?) thì báo lỗi sớm
            assert x.shape[0] in (1, 2, 4, 6, 8) or x.shape[0] > 1, "Chiều kênh không hợp lệ"
        else:
            raise RuntimeError(f"Sample có số chiều không hợp lệ: {x.shape}")
        return x.float().contiguous()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        inp = self._get_sample(self._inputs, idx)
        tgt = self._get_sample(self._targets, idx)
        item = {
            "input": inp,          # (1, 2F, T)
            "target": tgt,         # (1, 2F, T)
            "frequencies": self.frequencies,  # dùng chung cho mọi mẫu
            "times": self.times,
        }
        if self._meta is not None:
            item["meta"] = self._meta[idx]
        return item
