import torch
import numpy as np
import cv2
from collections import deque
from PIL import Image
import json
import os
import math

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu 


model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 448 


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
        T.ToTensor(), 
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# --- InternVideo Action Recognizer ---

class ActionRecognizer:
    def __init__(self, device='cuda:7', model_path=model_path, window_size=32):
        
        self.device = device
        self.window_size = window_size 
        self.buffers = {} # {track_id: deque[RGB PIL Image]}
        self.input_size = DEFAULT_INPUT_SIZE
        self.model_path = model_path
        model_path_str = str(model_path)
        if "MiniCPM" in model_path_str or "minicpm" in model_path_str:
            self.backend = "minicpm"
        elif "Mobile-VideoGPT" in model_path_str:
            self.backend = "mobilevideogpt"
        else:
            self.backend = "internvideo"

        self.target_labels = {
            "walking", "standing", "sitting on chair", "squat", "stretching",
            "eating", "drinking", "talking", "yarning",
            "cleaning floor", "reading book", "writing", "playing smart phone", "exercise", "massaging", "Unknown"
        }
        self.PROMPT_TEMPLATE = self._build_prompt_template()

        if self.backend == "minicpm":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(device).eval()
            self.transform = None
            print(f"ActionRecognizer (MiniCPM-V) initialized on {device}.")
        elif self.backend == "mobilevideogpt":
            try:
                from mobilevideogpt.utils import preprocess_input
            except Exception:
                self.backend = "internvideo"
                self.backend_fallback = True
                self.mv_preprocess_input = None
            if self.backend == "mobilevideogpt":
                self.mv_preprocess_input = preprocess_input
                config = AutoConfig.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=torch.float16,
                ).to(device).eval()
                self.transform = None
                print(f"ActionRecognizer (Mobile-VideoGPT) initialized on {device}.")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B", trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    "OpenGVLab/InternVideo2_5_Chat_8B",
                    trust_remote_code=True,
                ).half().to(torch.bfloat16).to(device).eval()
                self.transform = build_transform(input_size=self.input_size)
                print("Mobile-VideoGPT not available; falling back to InternVideo2.5.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(torch.bfloat16).to(device).eval()
            self.transform = build_transform(input_size=self.input_size)
            print(f"ActionRecognizer (InternVideo2.5) initialized on {device}.")


    def _build_prompt_template(self):
        """Construct Prompt"""
        action_list = ", ".join(sorted(list(self.target_labels)))
        

        prompt = f"""
        CONTEXT: This is a surveillance video clip of an indoor elderly care facility.
        TASK: Analyze the main action of the person in the video clip.
        INSTRUCTION: Only provide a JSON object. Describe the action that best fits the person in the scene.
        
        JSON: {{ "Action": "<short action descrption>"}}
        """
        return prompt

    def _query_mllm(self, frames_list):
        """
        Call InternVideo2.5 MLLM
        frames_list: List[PIL Image]
        """
        if self.backend == "minicpm":
            return self._query_minicpm(frames_list)
        if self.backend == "mobilevideogpt":
            return self._query_mobile_videogpt(frames_list)

        pixel_values_list, num_patches_list = [], []

        indices = np.linspace(0, len(frames_list) - 1, self.window_size).astype(int)
        
        for i in indices:
            img = frames_list[i] # PIL Image
            

            processed_images = dynamic_preprocess(img, image_size=self.input_size, use_thumbnail=True, max_num=1)
            

            pixel_values = [self.transform(tile) for tile in processed_images]
            pixel_values = torch.stack(pixel_values)
            
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        

        pixel_values = torch.cat(pixel_values_list)
        pixel_values = pixel_values.to(self.model.dtype).to(self.model.device)
        

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        question = video_prefix + self.PROMPT_TEMPLATE # 完整的问答输入
        

        generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1
        )
        

        with torch.no_grad():
            output_text, _ = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                question, 
                generation_config, 
                num_patches_list=num_patches_list, 
                history=None, 
                return_history=True
            )
        

        try:

            start_index = output_text.find('{')
            end_index = output_text.rfind('}') + 1
            json_str = output_text[start_index:end_index]
            
            result = json.loads(json_str)
            return result.get("Action", "Unknown").lower()
            
        except (ValueError, json.JSONDecodeError, IndexError):

            print(f"Parsing Failed. Raw MLLM output: {output_text[:100]}")
            return "Unknown"

    def _query_minicpm(self, frames_list):
        import numpy as np
        if not frames_list:
            return "Unknown"
        num = min(8, len(frames_list))
        indices = np.linspace(0, len(frames_list) - 1, num).astype(int)
        frames = [frames_list[i] for i in indices]
        try:
            if hasattr(self.model, "chat"):
                msgs = [{"role": "user", "content": self.PROMPT_TEMPLATE}]
                try:
                    output_text = self.model.chat(self.tokenizer, frames, self.PROMPT_TEMPLATE)
                except TypeError:
                    try:
                        output_text = self.model.chat(msgs, images=frames, tokenizer=self.tokenizer)
                    except TypeError:
                        output_text = self.model.chat(self.tokenizer, msgs, images=frames)
            else:
                return "Unknown"
        except Exception as e:
            print(f"MiniCPM inference failed: {e}")
            return "Unknown"

        try:
            start_index = output_text.find('{')
            end_index = output_text.rfind('}') + 1
            json_str = output_text[start_index:end_index]
            result = json.loads(json_str)
            return result.get("Action", "Unknown").lower()
        except Exception:
            return "Unknown"
    def _query_mobile_videogpt(self, frames_list):
        import tempfile
        import imageio.v3 as iio
        import numpy as np
        video_path = None
        try:
            frames_np = [np.array(f) for f in frames_list]
            fd, video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            iio.imwrite(video_path, frames_np, fps=8)
            input_ids, video_frames, context_frames, stop_str = self.mv_preprocess_input(
                self.model, self.tokenizer, video_path, self.PROMPT_TEMPLATE
            )
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=torch.stack(video_frames, dim=0).half().to(self.device),
                    context_images=torch.stack(context_frames, dim=0).half().to(self.device),
                    do_sample=False,
                    temperature=0,
                    top_p=1,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True,
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

        try:
            start_index = outputs.find('{')
            end_index = outputs.rfind('}') + 1
            json_str = outputs[start_index:end_index]
            result = json.loads(json_str)
            return result.get("Action", "Unknown").lower()
        except Exception:
            return "Unknown"

    def predict(self, track_id, crop_img, enable_infer=True):
        """
        Updates the buffer (PIL Images) and triggers MLLM reasoning when the window is full.
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.window_size)
            

        rgb_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.buffers[track_id].append(pil_img)
        

        K = 32 
        if len(self.buffers[track_id]) == self.window_size and enable_infer:
            
            # MLLM Inference Trigger
            final_label = self._query_mllm(list(self.buffers[track_id]))
            
            # 3. Clean Buffer (Sliding Window: remove K frames)
            for _ in range(K):
                if self.buffers[track_id]:
                    self.buffers[track_id].popleft() 
                
            return final_label
        
        return "buffering"

    def predict_clip(self, frames_list):
        """
        Direct clip-level action prediction from a list of PIL Images.
        """
        if not frames_list:
            return "Unknown"
        return self._query_mllm(frames_list)

    def clean(self, active_ids):
        """Cleans up buffers for lost track IDs."""
        for tid in list(self.buffers.keys()):
            if tid not in active_ids:
                del self.buffers[tid]
