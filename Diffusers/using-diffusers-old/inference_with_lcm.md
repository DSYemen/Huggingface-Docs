# نموذج الاتساق الكامن

تمكّن نماذج الاتساق الكامنة (LCMs) من إنشاء صور عالية الجودة بسرعة عن طريق التنبؤ المباشر بعملية الانتشار العكسي في الفراغ الكامن بدلاً من مساحة البكسل. وبعبارة أخرى، تحاول نماذج LCM التنبؤ بالصورة الخالية من الضوضاء من الصورة المشوشة، على عكس نماذج الانتشار النموذجية التي تزيل الضوضاء بشكل تكراري من الصورة المشوشة. ومن خلال تجنب عملية أخذ العينات التكرارية، يمكن لنماذج LCM إنشاء صور عالية الجودة في 2-4 خطوات بدلاً من 20-30 خطوة.

يتم استخلاص نماذج LCM من النماذج المُدربة مسبقًا والتي تتطلب حوالي 32 ساعة من الحوسبة A100. ولتسريع ذلك، تقوم LCM-LoRAs بتدريب مهايئ LoRA الذي يحتوي على عدد أقل بكثير من المعلمات التي يجب تدريبها مقارنة بالنموذج الكامل. يمكن توصيل LCM-LoRA بنموذج الانتشار بمجرد تدريبه.

سيوضح هذا الدليل كيفية استخدام نماذج LCM وLCM-LoRAs للاستدلال السريع على المهام، وكيفية استخدامها مع المهايئات الأخرى مثل ControlNet أو T2I-Adapter.

> [!TIP]
> تتوفر نماذج LCM وLCM-LoRAs لـ Stable Diffusion v1.5، وStable Diffusion XL، ونموذج SSD-1B. يمكنك العثور على نقاط التفتيش الخاصة بهم في [مجموعات الاتساق الكامنة](https://hf.co/collections/latent-consistency/latent-consistency-models-weights-654ce61a95edd6dffccef6a8).

## نص إلى صورة

<hfoptions id="lcm-text2img">
<hfoption id="LCM">

لاستخدام نماذج LCM، تحتاج إلى تحميل نقطة تفتيش LCM للنموذج المدعوم في [`UNet2DConditionModel`] واستبدال المخطط بـ [`LCMScheduler`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه نصي لإنشاء صورة في 4 خطوات فقط.

هناك بعض الملاحظات التي يجب مراعاتها عند استخدام نماذج LCM:

- عادةً ما يتم مضاعفة حجم الدفعة داخل الأنبوب للإرشاد الخالي من التصنيف. ولكن تطبق LCM الإرشاد باستخدام ترميزات الإرشاد ولا تحتاج إلى مضاعفة حجم الدفعة، مما يؤدي إلى استدلال أسرع. الجانب السلبي هو أن المطالبات السلبية لا تعمل مع LCM لأنها لا تؤثر على عملية إزالة التشويش.
- النطاق المثالي لـ `guidance_scale` هو [3.، 13.] لأن هذا ما تم تدريب UNet عليه. ومع ذلك، فإن تعطيل `guidance_scale` بقيمة 1.0 فعال أيضًا في معظم الحالات.

```python
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
"latent-consistency/lcm-sdxl",
torch_dtype=torch.float16,
variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(0)
image = pipe(
prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdxl_t2i.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

لاستخدام LCM-LoRAs، تحتاج إلى استبدال المخطط بـ [`LCMScheduler`] وتحميل أوزان LCM-LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه نصي لإنشاء صورة في 4 خطوات فقط.

هناك بعض الملاحظات التي يجب مراعاتها عند استخدام LCM-LoRAs:

- عادةً ما يتم مضاعفة حجم الدفعة داخل الأنبوب للإرشاد الخالي من التصنيف. ولكن تطبق LCM الإرشاد باستخدام ترميزات الإرشاد ولا تحتاج إلى مضاعفة حجم الدفعة، مما يؤدي إلى استدلال أسرع. الجانب السلبي هو أن المطالبات السلبية لا تعمل مع LCM لأنها لا تؤثر على عملية إزالة التشويش.
- يمكنك استخدام الإرشاد مع LCM-LoRAs، ولكنه حساس جدًا لقيم `guidance_scale` العالية ويمكن أن يؤدي إلى تشوهات في الصورة المولدة. أفضل القيم التي وجدناها هي بين [1.0، 2.0].
- استبدل [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0) بأي نموذج مُدرب بشكل دقيق. على سبيل المثال، جرب استخدام نقطة تفتيش [animagine-xl](https://huggingface.co/Linaqruf/animagine-xl) لإنشاء صور أنيمي باستخدام SDXL.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
variant="fp16",
torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
generator = torch.manual_seed(42)
image = pipe(
prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=1.0
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdxl_t2i.png"/>
</div>

</hfoption>
</hfoptions>

## صورة إلى صورة

<hfoptions id="lcm-img2img">
<hfoption id="LCM">

لاستخدام نماذج LCM للصورة إلى صورة، تحتاج إلى تحميل نقطة تفتيش LCM للنموذج المدعوم في [`UNet2DConditionModel`] واستبدال المخطط بـ [`LCMScheduler`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه نصي وصورة أولية لإنشاء صورة في 4 خطوات فقط.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps`، و`strength`، و`guidance_scale` للحصول على أفضل النتائج.

```python
import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image

unet = UNet2DConditionModel.from_pretrained(
"SimianLuo/LCM_Dreamshaper_v7",
subfolder="unet",
torch_dtype=torch.float16,
)

pipe = AutoPipelineForImage2Image.from_pretrained(
"Lykon/dreamshaper-7",
unet=unet,
torch_dtype=torch.float16,
variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
generator = torch.manual_seed(0)
image = pipe(
prompt,
image=init_image,
num_inference_steps=4,
guidance_scale=7.5,
strength=0.5,
generator=generator
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-img2img.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

لاستخدام LCM-LoRAs للصورة إلى الصورة، تحتاج إلى استبدال المخطط بـ [`LCMScheduler`] وتحميل أوزان LCM-LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. بعد ذلك، يمكنك استخدام الأنابيب كالمعتاد، وإمرار موجه نصي وصورة أولية لإنشاء صورة في 4 خطوات فقط.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps`، و`strength`، و`guidance_scale` للحصول على أفضل النتائج.

```py
import torch
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from diffusers.utils import make_image_grid, load_image

pipe = AutoPipelineForImage2Image.from_pretrained(
"Lykon/dreamshaper-7",
torch_dtype=torch.float16,
variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

generator = torch.manual_seed(0)
image = pipe(
prompt,
image=init_image,
num_inference_steps=4,
guidance_scale=1,
strength=0.6,
generator=generator
).images[0]
image
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-img2img.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>
</hfoptions>
## Inpainting

لاستخدام LCM-LoRAs للحشو، تحتاج إلى استبدال المُجدول ب [`LCMScheduler`] وتحميل أوزان LCM-LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. بعد ذلك، يمكنك استخدام الأنبوب كالمعتاد، وتمرير مُطَوِّر نصي وصورة أولية وصورة قناع لتوليد صورة في 4 خطوات فقط.

<div class="flex gap-4">
 <div>
 <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
 <figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
 </div>
 <div>
 <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm-lora-inpaint.png"/>
 <figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
 </div>
</div>

## المهايئات

تتوافق LCMs مع المهايئات مثل LoRA وControlNet وT2I-Adapter وAnimateDiff. يمكنك جلب سرعة LCMs إلى هذه المهايئات لتوليد الصور بأسلوب معين أو ضبط النموذج على إدخال آخر مثل صورة Canny.

### LoRA

يمكن ضبط مهايئات [LoRA](../using-diffusers/loading_adapters#lora) بسرعة على تعلم أسلوب جديد من بضع صور فقط وإضافتها إلى نموذج مُدرب مسبقًا لتوليد الصور بهذا الأسلوب.

<hfoptions id="lcm-lora">
<hfoption id="LCM">

قم بتحميل نقطة تفتيش LCM لنموذج المدعوم في [`UNet2DConditionModel`] واستبدل المُجدول ب [`LCMScheduler`]. بعد ذلك، يمكنك استخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LoRA في LCM وتوليد صورة ذات أسلوب في بضع خطوات.

```python
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained(
"latent-consistency/lcm-sdxl",
torch_dtype=torch.float16,
variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

prompt = "papercut, a cute fox"
generator = torch.manual_seed(0)
image = pipe(
prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdx_lora_mix.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

استبدل المُجدول ب [`LCMScheduler`]. بعد ذلك، يمكنك استخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LCM-LoRA وLoRA الأسلوب الذي تريد استخدامه. قم بدمج كلا مهايئي LoRA باستخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] وقم بتوليد صورة ذات أسلوب في بضع خطوات.

```py
import torch
from diffusers import DiffusionPipeline, LCMScheduler

pipe = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
variant="fp16",
torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])

prompt = "papercut, a cute fox"
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=4, guidance_scale=1, generator=generator).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdx_lora_mix.png"/>
</div>

</hfoption>
</hfoptions>

### ControlNet

[ControlNet](./controlnet) عبارة عن مهايئات يمكن تدريبها على مجموعة متنوعة من الإدخالات مثل حافة Canny أو تقدير الوضع أو العمق. يمكن إدراج ControlNet في الأنبوب لتوفير المزيد من الضبط والتحكم للنموذج من أجل التوليد الأكثر دقة.

يمكنك العثور على نماذج ControlNet الإضافية المدربة على إدخالات أخرى في مستودع [lllyasviel](https://hf.co/lllyasviel).

<hfoptions id="lcm-controlnet">
<hfoption id="LCM">

قم بتحميل نموذج ControlNet المدرب على صور Canny ومرره إلى [`ControlNetModel`]. بعد ذلك، يمكنك تحميل نموذج LCM في [`StableDiffusionControlNetPipeline`] واستبدال المُجدول ب [`LCMScheduler`]. الآن قم بتمرير صورة Canny إلى الأنبوب وقم بتوليد صورة.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps` و`controlnet_conditioning_scale` و`cross_attention_kwargs` و`guidance_scale` للحصول على أفضل النتائج.

```python
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image, make_image_grid

image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"SimianLuo/LCM_Dreamshaper_v7",
controlnet=controlnet,
torch_dtype=torch.float16,
safety_checker=None,
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)
image = pipe(
"the mona lisa",
image=canny_image,
num_inference_steps=4,
generator=generator,
).images[0]
make_image_grid([canny_image, image], rows=1, cols=2)
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_full_sdv1-5_controlnet.png"/>
</div>

</hfoption>
<hfoption id="LCM-LoRA">

قم بتحميل نموذج ControlNet المدرب على صور Canny ومرره إلى [`ControlNetModel`]. بعد ذلك، يمكنك تحميل نموذج Stable Diffusion v1.5 في [`StableDiffusionControlNetPipeline`] واستبدال المُجدول ب [`LCMScheduler`]. استخدم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان LCM-LoRA، ومرر صورة Canny إلى الأنبوب وقم بتوليد صورة.

> [!TIP]
> جرب قيمًا مختلفة لـ `num_inference_steps` و`controlnet_conditioning_scale` و`cross_attention_kwargs` و`guidance_scale` للحصول على أفضل النتائج.

```py
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
from diffusers.utils import load_image

image = load_image(
"https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
).resize((512, 512))

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
controlnet=controlnet,
torch_dtype=torch.float16,
safety_checker=None,
variant="fp16"
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

generator = torch.manual_seed(0)
image = pipe(
"the mona lisa",
image=canny_image,
num_inference_steps=4,
guidance_scale=1.5,
controlnet_conditioning_scale=0.8,
cross_attention_kwargs={"scale": 1},
generator=generator,
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lcm/lcm_sdv1-5_controlnet.png"/>
</div>

</hfoption>
</hfoptions>
بالتأكيد، سأقوم بترجمة النص الموجود في الفقرات والعناوين مع اتباع التعليمات التي قدمتها.

### T2I-Adapter
يوفر T2I-Adapter مدخلاً إضافياً لتشكيل نموذج مُدرب مسبقًا، وهو أخف وزنًا من ControlNet. إنه أسرع من ControlNet، ولكن النتائج قد تكون أسوأ قليلاً.

يمكنك العثور على نقاط تفتيش T2I-Adapter الإضافية المدربة على مدخلات أخرى في مستودع [TencentArc's].

### AnimateDiff
AnimateDiff هو محول يضيف الحركة إلى صورة. يمكن استخدامه مع معظم نماذج Stable Diffusion، مما يحولها فعليًا إلى نماذج "توليد الفيديو". يتطلب الحصول على نتائج جيدة باستخدام نموذج فيديو عادةً إنشاء عدة إطارات (16-24)، والتي يمكن أن تكون بطيئة جدًا مع نموذج Stable Diffusion العادي. يمكن لـ LCM-LoRA تسريع هذه العملية عن طريق إجراء 4-8 خطوات فقط لكل إطار.

قم بتحميل [`AnimateDiffPipeline`] ومرر [`MotionAdapter`] إليه. ثم استبدل الجدولة بـ [`LCMScheduler`]، وقم بدمج كلا محولات LoRA باستخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.set_adapters`]. الآن يمكنك تمرير موجه إلى الأنبوب وإنشاء صورة متحركة.