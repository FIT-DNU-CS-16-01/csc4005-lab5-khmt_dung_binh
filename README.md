[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kUEG02mW)
# CSC4005 Lab 5 – Vision Transformer for Smart Campus Scene Classification

## 1. Case study

**Smart Campus Scene Classification with Vision Transformer**

Hệ thống Smart Campus nhận ảnh từ camera hoặc thiết bị quan sát trong trường. Mô hình cần phân loại ảnh vào một trong các loại không gian gần với môi trường đại học:

```text
classroom
computerroom
library
corridor
office
```

Bộ dữ liệu được sử dụng là **MIT Indoor Scenes 67**. Repo này không chứa dữ liệu gốc. Sinh viên cần tải dữ liệu từ nguồn chính thức và truyền đường dẫn qua `--data_dir`.

## 2. Cấu trúc repo

```text
csc4005-lab5-khmt_dung_binh/
├── README.md
├── REPORT_TEMPLATE.md
├── requirements.txt
├── configs/
│   ├── baseline_vit_head_only.json
│   └── debug_smoke.json
├── docs/
│   ├── DATASET_GUIDE.md
│   ├── RUBRIC.md
│   ├── LAB_GUIDE_LAB5.md
│   └── WANDB_GUIDE.md
├── outputs/
│   ├── vit_b16_head_only/
│   │   ├── best_model.pt
│   │   ├── history.csv
│   │   ├── metrics.json
│   │   ├── curves.png
│   │   ├── confusion_matrix.png
│   │   ├── class_to_idx.json
│   │   └── config.json
│   └── vit_b16_finetune/
│       ├── best_model.pt
│       ├── history.csv
│       ├── metrics.json
│       ├── curves.png
│       ├── confusion_matrix.png
│       ├── class_to_idx.json
│       └── config.json
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── prepare_subset.py
│   ├── train.py
│   └── utils.py
├── ci/
│   ├── check_structure.py
│   └── smoke_train.py
└── .github/
    └── workflows/
        └── ci.yml
```

## 3. Cài đặt

Sử dụng conda environment đã có sẵn:

```bash
conda activate deep_learning
pip install -r requirements.txt
```

## 4. Chuẩn bị dữ liệu

### Tải bộ dữ liệu MIT Indoor Scenes 67

Tải và giải nén bộ dữ liệu MIT Indoor Scenes 67 vào thư mục `data/Images/`.

### Tạo subset 5 lớp

```bash
python -m src.prepare_subset \
  --source_dir "data/Images" \
  --output_dir "data/mit_indoor_smartcampus_5" \
  --classes classroom computerroom library corridor office \
  --max_per_class 400
```

Kết quả:
```text
classroom: 113 ảnh
computerroom: 114 ảnh
library: 107 ảnh
corridor: 346 ảnh
office: 109 ảnh
Tổng: 789 ảnh
```

## 5. Chạy thí nghiệm

### 5.1 Debug smoke test (kiểm tra pipeline)

```bash
python -m src.train \
  --config configs/debug_smoke.json \
  --data_dir "data/mit_indoor_smartcampus_5"
```

### 5.2 Baseline: ViT-B/16 head_only

```bash
python -m src.train \
  --config configs/baseline_vit_head_only.json \
  --data_dir "data/mit_indoor_smartcampus_5" \
  --run_name vit_b16_head_only
```

### 5.3 Mở rộng: ViT-B/16 finetune

```bash
python -m src.train \
  --data_dir "data/mit_indoor_smartcampus_5" \
  --project csc4005-lab6-mit-indoor-vit \
  --run_name vit_b16_finetune \
  --model_name vit_b_16 \
  --train_mode finetune \
  --epochs 5 \
  --batch_size 8 \
  --img_size 224 \
  --lr 0.00005 \
  --weight_decay 0.0001 \
  --dropout 0.2 \
  --augment \
  --use_wandb
```

## 6. Kết quả

### So sánh head_only vs finetune

| Metric | head_only (Test) | finetune (Test) |
|---|---:|---:|
| Accuracy | 91.38% | 95.69% |
| Macro-F1 | 88.11% | 93.85% |
| Trainable params | 3,845 (0.0045%) | 85,802,501 (100%) |
| Epoch time | ~20s | ~28s |

### Outputs

Mỗi run tạo thư mục `outputs/<run_name>/` chứa:

```text
best_model.pt
history.csv
metrics.json
curves.png
confusion_matrix.png
class_to_idx.json
config.json
```

## 7. W&B Dashboard

- Project: [csc4005-lab6-mit-indoor-vit](https://wandb.ai/models-dai-nam-university/csc4005-lab6-mit-indoor-vit)
- Run head_only: [vit_b16_head_only](https://wandb.ai/models-dai-nam-university/csc4005-lab6-mit-indoor-vit/runs/um106sli)
- Run finetune: [vit_b16_finetune](https://wandb.ai/models-dai-nam-university/csc4005-lab6-mit-indoor-vit/runs/k7rnpplj)

## 8. Checklist nộp bài

- [x] Có subset 5 lớp từ MIT Indoor Scenes 67
- [x] Chạy được ViT baseline ở chế độ `head_only`
- [x] Có W&B dashboard
- [x] Có `metrics.json`
- [x] Có `curves.png`
- [x] Có `confusion_matrix.png`
- [x] Có báo cáo theo `REPORT_TEMPLATE.md`
- [x] Có nhận xét lỗi mô hình và đề xuất cải thiện
- [x] So sánh head_only và finetune (bài mở rộng)
