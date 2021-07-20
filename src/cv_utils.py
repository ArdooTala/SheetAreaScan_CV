import numpy as np
import cv2
import pathlib


class SheetScanner:
    command_list = [
        "[  >> Decrease parameter",
        "]  >> Increase parameter",
        "SPACE  >> OK",
        "",
        "Output File  >> ./Polygons.txt",
        "", "",
        "Press any key to start . . ."
    ]

    def __init__(self, path: pathlib.Path, exp_dst: pathlib.Path):
        self.exp_dst = exp_dst
        self.img = cv2.imread(str(path), 1)
        cv2.namedWindow("KIR", cv2.WINDOW_NORMAL)

    def instructions(self):
        _img = self.img.copy()
        for i, inst in enumerate(SheetScanner.command_list):
            cv2.putText(_img, inst, (40, (i * 40 + 50)), 0, 1, (0, 255, 0), 4)
        cv2.imshow("KIR", _img)
        cv2.waitKey(0)

    def houghline_draw(self, img, v):
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, v, minLineLength=50, maxLineGap=1000)
        if lines is not None:
            cv2.putText(_img := self.img.copy(), "#Lines >> {}".format(len(lines)), (20, 200), 0, 2, (0, 255, 0))
            for line in lines:
                i = line[0]
                cv2.line(_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 255), 3)
            return _img

    def tweak(self, f, val, step, dst, min_val=None, max_val=None):
        cv2.imshow("KIR", img := f(val))
        while (k := cv2.waitKey(0)) != 32:
            if k == 93:
                val += step
            elif k == 91:
                val -= step

            if min_val is not None:
                val = max(val, min_val)
            if max_val is not None:
                val = min(val, max_val)

            img = f(val)
            _img = cv2.addWeighted(cv2.cvtColor(img.copy(), 1), 0.9, self.img, 0.1, 0.0)
            cv2.putText(_img, "{} >> {}".format(dst.stem, val), (20, 100), 0, 2, (0, 255, 0))
            cv2.imshow("KIR", _img)

        cv2.imwrite(str(dst), img)
        return img

    def refine(self):
        img = self.img.copy()

        gray = img[:, :, 2]

        blurred = self.tweak(lambda x: cv2.GaussianBlur(gray, (x, x), 0),
                             15, 2, self.exp_dst / "1-blurred.png", min_val=3)
        thrshld = self.tweak(lambda x: cv2.threshold(blurred, x, 255, cv2.THRESH_BINARY)[1],
                             140, 1, self.exp_dst / "2-Binary.png", min_val=0, max_val=255)
        cleaned = self.tweak(lambda x: cv2.dilate(cv2.erode(thrshld, None, iterations=x), None, iterations=x),
                             3, 1, self.exp_dst / "3-Cleaned.png", min_val=0)
        patched = self.tweak(lambda x: cv2.erode(cv2.dilate(cleaned, None, iterations=x), None, iterations=x),
                             3, 1, self.exp_dst / "4-Patched.png", min_val=0)
        margins = self.tweak(lambda x: cv2.erode(patched, None, iterations=x),
                             1, 1, self.exp_dst / "5-Safe_Margin.png", min_val=0)

        contours, hierarchy = cv2.findContours(
            margins,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)

        outline, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        outline = sorted(outline, key=lambda x: cv2.contourArea(x), reverse=True)

        cv2.drawContours(img, outline, 0, (0, 0, 255), 3)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        cv2.imshow("KIR", img)
        cv2.waitKey(0)

        cv2.imwrite(str(self.exp_dst / "Final.png"), img)
        with open(self.exp_dst / "Polygons.txt", 'w') as f:
            for cnt in contours:
                for vrt in cnt:
                    f.write(str(vrt))
                    f.write("\n")
                f.write("\n")

    def corners(self):
        pass
        # box = cv2.minAreaRect(outline[0])
        # corners = cv2.boxPoints(box)
        # for c in np.array(corners, dtype="int"):
        #     cv2.drawMarker(img, (c[0], c[1]), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=40)
        #     cv2.drawMarker(img, (c[0], c[1]), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=25, thickness=2)

        # dst = cv2.cornerHarris(thrshld, 20, 3, 0.05)
        # dst = cv2.dilate(dst, None, iterations=5)
        # dst = cv2.threshold(dst, 10, 255, cv2.THRESH_BINARY)[1]
        # cv2.drawContours(dst, outline, 0, 128, 1)
        # cv2.imshow("KIR", dst)
        # cv2.waitKey(0)
