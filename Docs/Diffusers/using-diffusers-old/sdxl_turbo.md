# Stable Diffusion XL Turbo

SDXL Turbo هو نموذج تنافسي مُقطَّر زمنيًا [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (SDXL) قادر على تشغيل الاستنتاج في خطوة واحدة فقط.

سيوضح هذا الدليل كيفية استخدام SDXL-Turbo للتحويل من نص إلى صورة ومن صورة إلى صورة.

قبل أن تبدأ، تأكد من تثبيت المكتبات التالية:

## تحميل نقاط تفتيش النموذج

قد تكون أوزان النموذج مخزنة في مجلدات فرعية منفصلة على Hub أو محليًا، وفي هذه الحالة، يجب عليك استخدام طريقة [`~StableDiffusionXLPipeline.from_pretrained`]:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("cuda")
```

يمكنك أيضًا استخدام طريقة [`~StableDiffusionXLPipeline.from_single_file`] لتحميل نقطة تفتيش النموذج المخزنة بتنسيق ملف واحد (`.ckpt` أو `.safetensors`) من Hub أو محليًا. بالنسبة لطريقة التحميل هذه، تحتاج إلى تعيين `timestep_spacing="trailing"` (يمكنك تجربة قيم الجدولة الأخرى للحصول على نتائج أفضل):

```py
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch

pipeline = StableDiffusionXLPipeline.from_single_file(
"https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors",
torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("cuda")
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
```

## من نص إلى صورة

بالنسبة للتحويل من نص إلى صورة، قم بتمرير موجه نصي. بشكل افتراضي، يقوم SDXL Turbo بإنشاء صورة بحجم 512x512، وهذا الحجم يعطي أفضل النتائج. يمكنك تجربة تعيين معلمات `height` و`width` إلى 768x768 أو 1024x1024، ولكن يجب أن تتوقع تدهور الجودة عند القيام بذلك.

تأكد من تعيين `guidance_scale` على 0.0 لتعطيله، حيث تم تدريب النموذج بدون ذلك. خطوة استنتاج واحدة كافية لتوليد صور عالية الجودة.

إن زيادة عدد الخطوات إلى 2 أو 3 أو 4 من شأنه أن يحسن جودة الصورة.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate Italian priest robe."

image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sdxl-turbo-text2img.png" alt="generated image of a racoon in a robe"/>
</div>

## من صورة إلى صورة

بالنسبة للتحويل من صورة إلى صورة، تأكد من أن `num_inference_steps * strength` أكبر من أو يساوي 1.

سينفذ خط أنابيب الصورة إلى الصورة `int(num_inference_steps * strength)` خطوات، على سبيل المثال `0.5 * 2.0 = 1` خطوة في مثالنا أدناه.

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline_image2image = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
init_image = init_image.resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipeline_image2image(prompt, image=init_image, strength=0.5, guidance_scale=0.0, num_inference_steps=2).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sdxl-turbo-img2img.png" alt="Image-to-image generation sample using SDXL Turbo"/>
</div>

## تسريع SDXL Turbo أكثر

- قم بتجميع UNet إذا كنت تستخدم إصدار PyTorch 2.0 أو أعلى. ستكون أول عملية استنتاج بطيئة جدًا، ولكن ستكون العمليات اللاحقة أسرع بكثير.

```py
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

- عند استخدام VAE الافتراضي، احتفظ به في `float32` لتجنب تحويلات `dtype` المكلفة قبل وبعد كل عملية توليد. تحتاج فقط إلى القيام بذلك مرة واحدة قبل أول عملية توليد:

```py
pipe.upcast_vae()
```

وكبديل لذلك، يمكنك أيضًا استخدام [VAE 16 بت](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) الذي أنشأه عضو المجتمع [`@madebyollin`](https://huggingface.co/madebyollin) والذي لا يحتاج إلى التحويل إلى `float32`.