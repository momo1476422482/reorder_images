from pathlib import Path
from typing import Dict, Tuple
from typing import List
import seaborn as sns

import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim
from yolo_test import YOLO


class chronology_images_dataset:
    """
    --divise the initial sequence into two classes :images with person/without person  [overlay_person,mask_person Vs
    without_person]
    --treatement of images with person : sort them by scene and then sort them by continuity of the
    person mask  [sorted_with_person Vs sorted_without_person]
    --treatement of images withoyt person sort them by scene
    --reconsturct the reordered sequence
    """

    # ==============================================================
    def __init__(self):

        self.model = YOLO()
        if not (Path(__file__).parent / 'sorted_with_person').is_dir():
            (Path(__file__).parent / 'sorted_with_person').mkdir()
        if not (Path(__file__).parent / 'sorted_without_person').is_dir():
            (Path(__file__).parent / 'sorted_without_person').mkdir()

    # ==============================================================
    @staticmethod
    def load_image(img_path: Path) -> np.ndarray:
        """
        load the image and transform it into grayscale one
        """
        img_color = cv2.imread(str(img_path))

        return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ==============================================================

    def get_features_2_images(self, ref_path: Path, img_path: Path) -> float:

        """
        get some matching features from the registration of 2 images
        :param ref: :param img: :return: homography transformation coefficient, number of matching inliners and the
        proportion of the overlapping area
        """
        img = self.load_image(img_path)
        ref = self.load_image(ref_path)
        return self.get_similarity_images(img, ref)

    # ===========================================================================================
    def sort_images_by_scene(
            self,
            path_imges: List[Path], is_person: bool = False
    ) -> Tuple[List[Path], List[float]]:
        """
        get matching features from a sequence of images (features are computed two /two)
        :param ref_path:
        :param img_path:
        :return:
        """
        ref_path = path_imges[0]
        list_res_final = []
        val_simi = []
        list_res_final.append(ref_path)
        i = 0
        while len(path_imges) != 1:
            list_res = []
            for idx, img_path in enumerate(path_imges):
                list_res.append(self.get_features_2_images(ref_path, img_path))
                print("ref", ref_path, "img", img_path)

            idx_ele_min = [list_res.index(i) for i in sorted(list_res)][1]
            print("list_res", list_res)
            val_simi.append(list_res[idx_ele_min])

            list_res_final.append(path_imges[idx_ele_min])

            ref_path_tmp = path_imges[idx_ele_min]
            if is_person is True:

                ref_number = ref_path_tmp.stem.split('_')[3]

                overlay_ref_path = Path(__file__).parent / 'overlay_person' / f'overlay_resized_frame_{ref_number}.png'
                mask_ref_path = Path(__file__).parent / 'mask_person' / f'mask_resized_frame_{ref_number}.png'

                cv2.imwrite(
                    str(Path(__file__).parent / 'sorted_with_person' / f"{i}th_img.png"),
                    cv2.imread(str(overlay_ref_path)),
                )
                cv2.imwrite(
                    str(Path(__file__).parent / 'sorted_with_person' / f"{i}th_mask_img.png"),
                    cv2.imread(str(mask_ref_path)),
                )
            else:

                cv2.imwrite(
                    f"{i}th_img.png",
                    cv2.imread(str(ref_path_tmp)),
                )
            path_imges.remove(ref_path)
            ref_path = ref_path_tmp
            i = i + 1
        print("list_simi", val_simi)

        return list_res_final

    # ==============================================================
    def combine_2_seq(self, list1: List[Path], list2: List[Path]) -> str:
        dict_res: Dict[str:float] = {}
        list1_end = self.load_image(list1[-1])
        list1_head = self.load_image(list1[0])
        list2_end = self.load_image(list2[-1])
        list2_head = self.load_image(list2[0])
        dict_res["1end2head"] = self.get_similarity_images(list1_end, list2_head)
        dict_res["1end2end"] = self.get_similarity_images(list1_end, list2_end)
        dict_res["1head2head"] = self.get_similarity_images(list1_head, list2_head)
        dict_res["1head2end"] = self.get_similarity_images(list1_head, list2_end)

        val_min = min(list(dict_res.values()))
        key_min = list(dict_res.keys())[list(dict_res.values()).index(val_min)]
        print("key_min", key_min, dict_res.values())
        return key_min

    # ==============================================================
    @staticmethod
    def get_segments(list_val: List[float], threshold: float) -> List[int]:
        res: List[int] = []
        for ind, ele in enumerate(list_val):
            if ele >= threshold and ind > 5:
                res.append(ind)
        return res

    # ==============================================================
    def get_list_val(self, list_path: List[Path]) -> List[float]:
        res: List[float] = []
        for ind, ele in enumerate(list_path):
            if ind == 0:
                continue
            else:
                res.append(
                    self.get_features_2_images(
                        list_path[ind],
                        list_path[ind - 1],
                    )
                )
        return res

    # ==============================================================
    @staticmethod
    def get_similarity_images(img1: np.ndarray, img2: np.ndarray, algo: str = "Euclidean"):
        if algo == "Euclidean":
            return np.sqrt(np.sum(np.square(img1.flatten() - img2.flatten())))
        if algo == "ssim":
            score = ssim(img1, img2)
            return 1 - score

    # ==========================================================================
    def create_list_segment(self, list_path: List[Path], threhsold: float) -> List[Path]:
        list_val=self.get_list_val(list_path)
        segment_points = self.get_segments(list_val, threhsold)
        print("seg_pt", segment_points)
        pt_init = 0
        list_segment = []

        i = 0
        for ele in segment_points:
            segment = list_path[pt_init: ele + 1]

            print(f"{i}th segment", pt_init, ele)
            list_segment.append(segment)

            pt_init = ele + 1
            i = i + 1
        if pt_init < len(list_val):
            list_segment.append(list_path[pt_init: len(list_val) + 1])
        return list_segment

    # ==========================================================================
    def sort_images_by_person(self, list_segment: List[Path]) -> List[Path]:
        res_final = list_segment[0]
        res_tmp = list_segment[0]
        list_segment.remove(res_tmp)
        len_list = len(list((Path(__file__).parent / "sorted_with_person").glob('*.png'))) / 2
        while len(res_final) != int(len_list):
            print("res_final_len", len(res_final))
            for segment in list_segment:
                print("segment", segment)
                res = self.combine_2_seq(res_final, segment)
                key_value = res

                if key_value == "1end2end":
                    res_final = res_final + segment[::-1]
                    list_segment.remove(segment)
                elif key_value == "1head2head":
                    res_final = res_final[::-1] + segment
                    list_segment.remove(segment)
                elif key_value == "1head2end":
                    res_final = res_final[::-1] + segment[::-1]
                    list_segment.remove(segment)
                else:
                    res_final = res_final + segment
                    list_segment.remove(segment)
        return res_final

    # ======================================================================
    def reconstruct_video(self, res_final: List[Path], is_person: bool = False):

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(Path(__file__).parent/"video.mp4"), fourcc, 1, (672, 224), True)

        for i, ele in enumerate(res_final[::-1]):
            print(ele)
            name_img = ele.stem.split('_')[0] + '_' + ele.stem.split('_')[2]
            print(name_img)
            if is_person:
                cv2.imwrite(str(Path(__file__).parent/f"{i}th_img.png"),
                            cv2.imread(str(Path(__file__).parent / 'sorted_with_person' / f'{name_img}.png')))
                video.write(
                    cv2.imread(str(Path(__file__).parent / 'sorted_with_person' / f'{name_img}.png')).astype("uint8"))
                print("write to video!!")

        cv2.destroyAllWindows()
        video.release()

    # ===================================================================
    def run(self, threshold: float):
        list_path_img = []
        for img_path in list((Path(__file__).parent / "raw").glob("*.png")):
            list_path_img.append(img_path)
        # division des image avec/sans personne
        self.model.detect(list_path_img[0:20], saveas=Path(__file__).parent / 'person.csv')
        self.model.detect(list_path_img[20:50], saveas=Path(__file__).parent / 'perso1.csv')
        self.model.detect(list_path_img[50:80], saveas=Path(__file__).parent / 'perso1.csv')
        self.model.detect(list_path_img[80:120], saveas=Path(__file__).parent / 'perso1.csv')

        # treatment des images avec personne
        path_images = Path(__file__).parent / "overlay_person"
        path_init = Path(__file__).parent / "overlay_person/overlay_resized_frame_042.png"
        path_imges = [path_init]
        for p_img in list(path_images.glob("*.png")):
            if p_img == path_init:
                continue
            else:
                path_imges.append(p_img)
        self.sort_images_by_scene(path_imges, is_person=True)

        list_path_mask = []
        len_list = len(list((Path(__file__).parent / "sorted_with_person").glob('*.png'))) / 2
        for i in range(int(len_list)):
            mask_path = Path(__file__).parent / "sorted_with_person" / f"{i}th_mask_img.png"
            list_path_mask.append(mask_path)
        index = np.array(range(int(len_list) - 1))

        list_val = self.get_list_val(list_path_mask)
        print(list_val)
        plt.figure()
        plt.plot(index, np.asarray(list_val))
        plt.show()
        plt.savefig("rrrr.png")

        segment_points = self.get_segments(list_val, threshold)
        print("seg_pt", segment_points)
        list_segment = self.create_list_segment(list_path_mask, threshold)
        res_final_person = self.sort_images_by_person(list_segment)
        self.reconstruct_video(res_final_person,is_person=True)

        # treatment des images avec personne
        # reconstruct the whole sequence


# ===============================================================================================
if __name__ == "__main__":
    chrono = chronology_images_dataset()
    chrono.run(threshold=350)
