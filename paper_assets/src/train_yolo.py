from pathlib import Path

from ultralytics import YOLO


def run(model_size, task, project_path):
    if task == "segment":
        model = YOLO(f"yolo26{model_size}-seg.pt")
    else:
        model = YOLO(f"yolo26{model_size}.pt")

    batch_mapping = {"n": 24, "s": 24, "m": 12, "l": 10, "x": 8}
    model.train(
        data=project_path / "data/yolo_data/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=batch_mapping[model_size],
        name=f"{model_size}",
        project=project_path / "runs/" / task,
        exist_ok=True,
    )

    model = YOLO(project_path / f"runs/{task}/{model_size}/weights/best.pt")
    model.export(format="tensorrt", half=True, dynamic=False, batch=1)


if __name__ == "__main__":
    task = "segment"
    project_path = Path("/workspace/taco/")

    for model_size in ["n", "s", "m", "l", "x"]:
        run(model_size, task, project_path)
