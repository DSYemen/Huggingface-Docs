بالتأكيد، سألتزم بالتعليمات المذكورة في طلب الترجمة.

# T-GATE

يُسرع T-GATE عملية الاستنتاج لأنابيب Stable Diffusion وPixArt و Latency Consistency Model عن طريق تخطي حساب cross-attention بمجرد تقاربه. لا تتطلب هذه الطريقة أي تدريب إضافي ويمكن أن تسرع الاستنتاج بنسبة تتراوح بين 10-50%. T-GATE متوافق أيضًا مع طرق التحسين الأخرى مثل DeepCache.

قبل البدء، تأكد من تثبيت T-GATE.

للاستخدام T-GATE مع خط أنابيب، يجب استخدام محملها المقابل.

للتمكين T-GATE لخط أنابيب معين، قم بإنشاء `TgateLoader` باستخدام خط الأنابيب، وخطوة البوابة (خطوة الوقت لوقف حساب الانتباه المتقاطع)، وعدد خطوات الاستنتاج. ثم استدعاء طريقة `tgate` على خط الأنابيب باستخدام موجه، وخطوة البوابة، وعدد خطوات الاستنتاج.

دعونا نرى كيف يمكن تمكين هذا لعدة خطوط أنابيب مختلفة.

## المعايير

### PixArt

لتسريع `PixArtAlphaPipeline` باستخدام T-GATE:

```py
import torch
from diffusers import PixArtAlphaPipeline
from tgate import TgatePixArtLoader

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype = torch.float16)

خطوة البوابة = 8
خطوة الاستدلال = 25
الأنبوب = TgatePixArtLoader (
الأنبوب,
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال،
). إلى ("cuda")

الصورة = الأنابيب. tgate (
"ألباكا مصنوعة من كتل بناء ملونة، سايبربانك."،
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال،
). images [0]
```

### Stable Diffusion XL

لتسريع `StableDiffusionXLPipeline` باستخدام T-GATE:

```بايثون
استيراد الشعلة
من الناشرين استيراد StableDiffusionXLPipeline
من الناشرين استيراد DPMSolverMultistepScheduler
من tgate استيراد TgateSDXLLoader

الأنبوب = StableDiffusionXLPipeline.from_pretrained (
"stabilityai/stable-diffusion-xl-base-1.0"،
torch_dtype = torch.float16،
variant = "fp16"،
use_safetensors = True،
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config (pipe.scheduler.config)

خطوة البوابة = 10
خطوة الاستدلال = 25
الأنبوب = TgateSDXLLoader (
الأنبوب،
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال،
). إلى ("cuda")

الصورة = الأنابيب. tgate (
"رائد فضاء في الأدغال، لوحة الألوان الباردة، الألوان المكتومة، مفصلة، 8k."،
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال
). images [0]
```

### StableDiffusionXL with DeepCache

لتسريع `StableDiffusionXLPipeline` باستخدام DeepCache و T-GATE:

```بايثون
استيراد الشعلة
من الناشرين استيراد StableDiffusionXLPipeline
من الناشرين استيراد DPMSolverMultistepScheduler
من tgate استيراد TgateSDXLDeepCacheLoader

الأنبوب = StableDiffusionXLPipeline.from_pretrained (
"stabilityai/stable-diffusion-xl-base-1.0"،
torch_dtype = torch.float16،
variant = "fp16"،
use_safetensors = True،
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config (pipe.scheduler.config)

خطوة البوابة = 10
خطوة الاستدلال = 25
الأنبوب = TgateSDXLDeepCacheLoader (
الأنبوب،
cache_interval = 3،
cache_branch_id = 0،
). إلى ("cuda")

الصورة = الأنابيب. tgate (
"رائد فضاء في الأدغال، لوحة الألوان الباردة، الألوان المكتومة، مفصلة، 8k."،
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال
). images [0]
```

### Latent Consistency Model

لتسريع `latent-consistency/lcm-sdxl` باستخدام T-GATE:

```بايثون
استيراد الشعلة
من الناشرين استيراد StableDiffusionXLPipeline
من الناشرين استيراد UNet2DConditionModel، LCMScheduler
من الناشرين استيراد DPMSolverMultistepScheduler
من tgate استيراد TgateSDXLLoader

unet = UNet2DConditionModel.from_pretrained (
"latent-consistency/lcm-sdxl"،
torch_dtype = torch.float16،
variant = "fp16"،
)
الأنبوب = StableDiffusionXLPipeline.from_pretrained (
"stabilityai/stable-diffusion-xl-base-1.0"،
unet = unet،
torch_dtype = torch.float16،
variant = "fp16"،
)
pipe.scheduler = LCMScheduler.from_config (pipe.scheduler.config)

خطوة البوابة = 1
خطوة الاستدلال = 4
الأنبوب = TgateSDXLLoader (
الأنبوب،
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال،
lcm = True
). إلى ("cuda")

الصورة = الأنابيب. tgate (
"رائد فضاء في الأدغال، لوحة الألوان الباردة، الألوان المكتومة، مفصلة، 8k."،
gate_step = gate_step،
num_inference_steps = خطوة الاستدلال
). images [0]
```

يدعم T-GATE أيضًا [`StableDiffusionPipeline`] و [PixArt-alpha/PixArt-LCM-XL-2-1024-MS].

## المعايير

| النموذج | MACs | المعلمة | الكمون | الصفر-10K-FID على MS-COCO |
|-----------------------|----------|-----------|---------|---------------------------|
| SD-1.5 | 16.938T | 859.520M | 7.032s | 23.927 |
| SD-1.5 w/ T-GATE | 9.875T | 815.557M | 4.313s | 20.789 |
| SD-2.1 | 38.041T | 865.785M | 16.121s | 22.609 |
| SD-2.1 w/ T-GATE | 22.208T | 815.433 M | 9.878s | 19.940 |
| SD-XL | 149.438T | 2.570B | 53.187s | 24.628 |
| SD-XL w/ T-GATE | 84.438T | 2.024B | 27.932s | 22.738 |
| Pixart-Alpha | 107.031T | 611.350M | 61.502s | 38.669 |
| Pixart-Alpha w/ T-GATE | 65.318T | 462.585M | 37.867s | 35.825 |
| DeepCache (SD-XL) | 57.888T | - | 19.931s | 23.755 |
| DeepCache w/ T-GATE | 43.868T | - | 14.666s | 23.999 |
| LCM (SD-XL) | 11.955T | 2.570B | 3.805s | 25.044 |
| LCM w/ T-GATE | 11.171T | 2.024B | 3.533s | 25.028 |
| LCM (Pixart-Alpha) | 8.563T | 611.350M | 4.733s | 36.086 |
| LCM w/ T-GATE | 7.623T | 462.585M | 4.543s | 37.048 |

تم اختبار الكمون على NVIDIA 1080TI، وتم حساب MACs وParams باستخدام [calflops]، وتم حساب FID باستخدام [PytorchFID].