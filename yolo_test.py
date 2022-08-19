import torch

from typing import cast, List
import cv2
from pathlib import Path
import numpy as np
import pandas as pd


def load_image(img_path: Path) -> np.ndarray:
    """
    load the image and transform it into grayscale one
    """
    img_color = cv2.imread(str(img_path))

    return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


class YOLO:
    """
    YOLO object detector for person / no_person classification

    https://github.com/ultralytics/yolov5/issues/36
    """

    # ============================================================================
    def __init__(self) -> None:

        """
        https://github.com/ultralytics/yolov5#pretrained-checkpoints
        """

        self.model = torch.hub.load("ultralytics/yolov5", "YOLOv5l6".lower())

        self.size = 1280
        if not (Path(__file__).parent/'mask_person').is_dir():
            (Path(__file__).parent / 'mask_person').mkdir()
        if not (Path(__file__).parent/'overlay_person').is_dir():
            (Path(__file__).parent / 'overlay_person').mkdir()
        if not (Path(__file__).parent/'without_person').is_dir():
            (Path(__file__).parent / 'without_person').mkdir()


    # ============================================================================
    def detect(self, path2images: List[Path], saveas: Path) -> None:

        if path2images:

            imgs = list(
                map(lambda x: cast(np.ndarray, cv2.imread(str(x))), path2images)
            )
            res = self.model(imgs, size=self.size)

            records = []
            for k, (path, img) in enumerate(zip(path2images, imgs)):

                frame = res.pandas().xyxy[k]

                if not frame.empty:

                    frame["image"] = str(path)
                    records += frame.to_dict("records")
                    print(f"found {frame['name'].tolist()} on {path}")

                    people = frame.query("name == 'person'")
                    if not people.empty:

                        mask = np.zeros((img.shape[0], img.shape[1]))

                        for ix in range(people.shape[0]):
                            dic = people.iloc[ix]

                            xmin = int(dic["xmin"])
                            xmax = int(dic["xmax"])
                            ymin = int(dic["ymin"])
                            ymax = int(dic["ymax"])

                            mask[ymin:ymax, xmin:xmax] = 1

                        dest = Path(__file__).parent / "mask_person"/f"mask_{path.name}"
                        cv2.imwrite(str(dest), mask * 250)
                        print(f"person mask saved to {dest}")

                        img = img.astype(float)
                        dest = Path(__file__).parent / "overlay_person"/f"overlay_{path.name}"
                        img[:, :, 1] -= mask * 50
                        img[:, :, 2] -= mask * 50
                        cv2.imwrite(str(dest), img.clip(min=0))
                        print(f"person mask overlay saved to {dest}")

                else:
                    dest = Path(__file__).parent / "without_person" / f"{path.name}"
                    cv2.imwrite(str(dest), img.clip(min=0))

            pd.DataFrame(records).to_csv(saveas)

        else:

            print("no image provided for detection")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    list_path_img = []
    model_yolo = YOLO()
    for img_path in list((Path(__file__).parent/"raw").glob("*.png")):

        list_path_img.append(img_path)

    model_yolo.detect(list_path_img[0:20],saveas=Path(__file__).parent/'person.csv')
    model_yolo.detect(list_path_img[20:50], saveas=Path(__file__).parent / 'perso1.csv')
    model_yolo.detect(list_path_img[50:80], saveas=Path(__file__).parent / 'perso1.csv')
    model_yolo.detect(list_path_img[80:120], saveas=Path(__file__).parent / 'perso1.csv')






