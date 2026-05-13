from paddleocr import PaddleOCR
import numpy as np
import re  

class TimeOCR:
    def __init__(self, use_gpu=True):
        print("-> Initializing OCR Model (PaddleOCR)...")
        # lang='ch' provides better support for Chinese characters, though digits are universal
        self.ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=use_gpu, show_log=False)

    def parse_chinese_time(self, raw_text):
        """
        Converts '2019年07月18日 星期四 09:54:09' to '2019-07-18 09:54:09'
        """
        if not raw_text:
            return ""

        # Regex Logic:
        # (\d{4})   -> Extract 4-digit year
        # 年        -> Match Chinese character for 'Year'
        # (\d{1,2}) -> Extract 1-2 digit month
        # 月        -> Match Chinese character for 'Month'
        # (\d{1,2}) -> Extract 1-2 digit day
        # 日        -> Match Chinese character for 'Day'
        # .*?       -> Ignore weekday text like " 星期四 " (non-greedy match)
        # (\d{1,2}:\d{1,2}:\d{1,2}) -> Extract time portion HH:MM:SS
        pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日.*?(\d{1,2}:\d{1,2}:\d{1,2})"
        
        match = re.search(pattern, raw_text)
        if match:
            year, month, day, time_str = match.groups()
            # Reassemble and ensure month/day are two digits (zfill)
            # Example: 2019-7-1 -> 2019-07-01
            formatted_time = f"{year}-{month.zfill(2)}-{day.zfill(2)} {time_str}"
            return formatted_time
        
        # If regex fails (possibly due to formatting issues), return raw text as a fallback
        # Alternative approach: replace '年'/'月' with '-' and remove '日'
        # However, relying on the regex pattern above is recommended for accuracy
        return raw_text

    def recognize(self, full_frame, roi_bbox=None):
        """
        Perform OCR on the full frame or a specific Region of Interest (ROI).
        """
        if roi_bbox is not None:
            x1, y1, x2, y2 = map(int, roi_bbox)
            h, w = full_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            img_crop = full_frame[y1:y2, x1:x2]
        else:
            img_crop = full_frame

        if img_crop.size == 0:
            return ""

        result = self.ocr.ocr(img_crop, cls=False)
        
        txts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                # Simple filtering: keep lines containing '201' (year prefix) or ':' (time separator)
                if '201' in text or ':' in text:
                    txts.append(text)
        
        raw_str = " ".join(txts)
        
        # Clean and format the extracted text
        clean_time = self.parse_chinese_time(raw_str)
        
        return clean_time