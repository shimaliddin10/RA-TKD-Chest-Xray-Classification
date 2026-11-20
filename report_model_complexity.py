"""
Reports parameter counts and FLOPs for the teacher (DenseNet161) and
student (RA-TKD-DeiT) models, in the same units used in the CXR-KD
literature (Params in millions, FLOPs in GFLOPs at a single forward pass).


Requires: pip install thop


Author: Mohammed Shimal Iddin
Student Number: 40490740
Module: ECS8056: Themed Research Project (2251_SUM)
Supervisor: Dr. Muhammad Fahim
Last Updated: 2026-08-31
"""


import torch
from thop import profile, clever_format


from config import *
from model_teacher_densenet import DenseNet161TeacherWithFeatures
from novel_model import RATKDDeiT



def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



def report_model(name, model, input_size, device):
    model = model.to(device).eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)


    total_params, trainable_params = count_params(model)


    # thop.profile hooks most standard layers (Conv2d, Linear, BatchNorm, etc.)
    # For models with custom forward signatures (e.g. return_features kwarg),
    # thop calls model(dummy_input) with just the positional input, which
    # matches the default forward path for both models here.
    macs, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2  # MACs -> FLOPs (each MAC = 1 multiply + 1 add)


    macs_fmt, params_fmt = clever_format([macs, params_thop], "%.3f")
    flops_fmt, _ = clever_format([flops, params_thop], "%.3f")


    print(f"\n=== {name} ===")
    print(f"  Input size:            {input_size}x{input_size}")
    print(f"  Total parameters:      {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"  Trainable parameters:  {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    print(f"  MACs:                  {macs_fmt}")
    print(f"  FLOPs (2x MACs):       {flops_fmt}")


    return {
        "model": name,
        "params_M": round(total_params / 1e6, 3),
        "trainable_params_M": round(trainable_params / 1e6, 3),
        "flops_G": round(flops / 1e9, 3),
    }



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    teacher = DenseNet161TeacherWithFeatures(num_classes=NUM_CLASSES)
    student = RATKDDeiT(num_classes=NUM_CLASSES)


    results = []
    results.append(report_model("Teacher (DenseNet161)", teacher, IMAGE_SIZE, device))
    results.append(report_model("Student (RA-TKD-DeiT)", student, IMAGE_SIZE, device))


    print("\n=== Summary Table ===")
    print(f"{'Model':<25} {'Params (M)':<12} {'FLOPs (G)':<12}")
    for r in results:
        print(f"{r['model']:<25} {r['params_M']:<12} {r['flops_G']:<12}")


    import pandas as pd
    pd.DataFrame(results).to_csv("model_complexity_report.csv", index=False)
    print("\nSaved results to model_complexity_report.csv")



if __name__ == "__main__":
    main()